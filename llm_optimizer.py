import re
from typing import List, Dict, Tuple, Union
import numpy as np
from collections import deque


class LLM_optimizer:
    def __init__(self, creator_bot, lang='only in English', eval_len=130, improvement_len=200,
                 eval_temp=0.1, improve_temp=1.1, population_size=3, memory_size=5):
        self.creator_bot = creator_bot
        self.lang = lang
        self.improvement_len = improvement_len
        self.eval_len = eval_len
        self.eval_temp = eval_temp
        self.improve_temp = improve_temp
        self.result = {}
        self.population_size = population_size
        self.solution_memory = deque(maxlen=memory_size)  # Память последних успешных решений
        self.improvement_history = []  # История улучшений для адаптации параметров


    def evaluate_response(self, response, success_criteria, initial_task, n_tokens=100):
        feedback = []
        total_quality_score = 0
        num_criteria = len(success_criteria)

        for criterion in success_criteria:
            evaluation_prompt = f"Есть задача: {initial_task}\n\n Есть ответ:{response}\n\n Оцени следующий ответ по критерию: '{criterion}'. Какую оценку ты поставил бы от 0 до 100 и почему? Оценка ({self.lang}):"
            evaluation_result = self.creator_bot.create_text(evaluation_prompt, False, max_tokens=n_tokens, temp=self.eval_temp)

            # Предполагаем, что результат содержит и оценку, и пояснение
            # Разделим результат на оценку и объяснение

            # Используем регулярное выражение для поиска оценки
            score_match = re.search(r'(\d+(\.\d+)?)', evaluation_result)  # Ищем число с возможной десятичной точкой

            if score_match:
                score = float(score_match.group(0))  # Получаем найденное число
            else:
                score = 0.0  # По умолчанию, если нет совпадений, устанавливаем оценку в 0

            # Извлекаем критику, оставив оставшуюся часть строки
            critique = evaluation_result
            total_quality_score += score

            feedback.append({
                'criterion': criterion,
                'score': score,
                'critique': critique.strip()
            })

        # Рассчитываем среднюю оценку
        average_quality_score = total_quality_score / num_criteria if num_criteria > 0 else 0

        return {'quality': average_quality_score, 'feedback': feedback}

    def generate_diverse_solutions(self, initial_task: str, n: int) -> List[str]:
        """Генерация популяции разнообразных решений"""
        solutions = []
        prompts = [
            f"Дай креативное и нестандартное решение задачи: {initial_task}",
            f"Дай максимально простое и прямолинейное решение задачи: {initial_task}",
            f"Дай самое детальное и подробное решение задачи: {initial_task}",
            f"Попробуй решить задачу: {initial_task} используя аналогии из природы",
            f"Реши задачу: {initial_task} как если бы у тебя были неограниченные ресурсы"
        ]

        for i in range(n):
            prompt = prompts[i % len(prompts)]
            solution = self.creator_bot.create_text(prompt, False,
                                                    max_tokens=self.improvement_len,
                                                    temp=0.8 + (i * 0.2))  # Увеличиваем температуру для разнообразия
            solutions.append(self.creator_bot.filtration(solution))
        return solutions

    def crossover_solutions(self, solution1: str, solution2: str) -> str:
        """Скрещивание двух решений"""
        prompt = f"""Есть два решения одной задачи:
        Решение 1: {solution1}
        Решение 2: {solution2}

        Создай новое решение, объединив лучшие части обоих решений в одно целое.
        Ответ должен быть согласованным и логичным. ({self.lang}):"""

        return self.creator_bot.create_text(prompt, False,
                                            max_tokens=self.improvement_len,
                                            temp=0.5)

    def mutate_solution(self, solution: str, mutation_strength: float = 0.3) -> str:
        """Мутация решения с заданной силой"""
        prompt = f"""Измени следующее решение, сохранив его основную суть, 
        но добавив новые идеи или убрав лишнее. Сила изменений: {mutation_strength * 100}%

        Исходное решение: {solution}

        Дай измененную версию решения ({self.lang}):"""

        return self.creator_bot.create_text(prompt, False,
                                            max_tokens=self.improvement_len,
                                            temp=mutation_strength * 2)

    def analyze_improvement_pattern(self) -> Dict:
        """Анализ паттернов улучшения для адаптации параметров"""
        if len(self.improvement_history) < 2:
            return {
                'trend': 0,
                'volatility': 0,
                'plateau': False
            }

        scores = [x['quality'] for x in self.improvement_history]
        improvements = np.diff(scores)

        return {
            'trend': np.mean(improvements),
            'volatility': np.std(improvements),
            'plateau': len(improvements) > 3 and all(abs(i) < 1.0 for i in improvements[-3:])
        }

    def adaptive_temperature(self, pattern: Dict) -> float:
        """Адаптивная настройка температуры на основе паттерна улучшений"""
        base_temp = self.improve_temp
        if pattern.get('plateau', False):
            return base_temp * 1.5

        trend = pattern.get('trend', 0.0)
        if trend > 0:
            return base_temp * (1.0 - min(0.3, trend))
        return base_temp * (1.0 + min(0.5, abs(trend)))

    def improve_response(self, response, feedback, initial_task, n_tokens=220, temp=0.8):
        # Объединяем критику и создаем промт для улучшения
        critique_notes = "\n".join([f"Критика по критерию '{f['criterion']}': {f['critique']}" for f in feedback])

        improvement_prompt = f"Есть задача: {initial_task}\n\n Есть ответ:{response}\n\n Напиши лучший ответ с учетом этой критики (не пиши свою оценку ответа, только улучшенный ответ!):\n{critique_notes}\nРазмер поля ввода {int(n_tokens/3)} слов\nОтвет ({self.lang}):"
        improved_response = self.creator_bot.create_text(improvement_prompt, False, max_tokens=n_tokens, temp=temp)
        improved_response = self.creator_bot.filtration(improved_response)
        return improved_response

    def iterative_improvement(self, initial_task: str, success_criteria: List[str],
                              max_iterations: int = 10, desired_quality: float = 8.0) -> str:
        # Генерируем начальную популяцию решений
        population = self.generate_diverse_solutions(initial_task, self.population_size)
        best_result = {'quality': 0, 'value': '', 'feedback': []}

        for iteration in range(max_iterations):
            print(f"---------------- Итерация {iteration + 1} из {max_iterations} ----------------")

            # Оцениваем всю популяцию
            evaluated_population = []
            for solution in population:
                feedback = self.evaluate_response(solution, success_criteria, initial_task, self.eval_len)
                evaluated_population.append({
                    'solution': solution,
                    'quality': feedback['quality'],
                    'feedback': feedback['feedback']
                })

            # Сортируем по качеству
            evaluated_population.sort(key=lambda x: x['quality'], reverse=True)

            # Обновляем лучший результат
            if evaluated_population[0]['quality'] > best_result['quality']:
                best_result = {
                    'quality': evaluated_population[0]['quality'],
                    'value': evaluated_population[0]['solution'],
                    'feedback': evaluated_population[0]['feedback']
                }
                self.solution_memory.append(best_result['value'])
                self.improvement_history.append(best_result)
                print('UPDATE', best_result['quality'])
                print(best_result['value'])

            if best_result['quality'] >= desired_quality:
                print(f"Достигнуто желаемое качество: {best_result['quality']}.")
                break

            # Анализируем паттерн улучшений
            pattern = self.analyze_improvement_pattern()
            adaptive_temp = self.adaptive_temperature(pattern)

            # Формируем новую популяцию
            new_population = [evaluated_population[0]['solution']]  # Сохраняем лучшее решение

            # Добавляем скрещенные решения
            for i in range(min(len(evaluated_population) - 1, 2)):
                crossed = self.crossover_solutions(
                    evaluated_population[i]['solution'],
                    evaluated_population[i + 1]['solution']
                )
                new_population.append(crossed)

            # Добавляем мутированные версии лучших решений
            while len(new_population) < self.population_size:
                parent_solution = evaluated_population[np.random.randint(0, len(evaluated_population))]['solution']
                mutation_strength = float(np.random.uniform(0.1, 0.5))
                mutated = self.mutate_solution(parent_solution, mutation_strength)
                new_population.append(mutated)

            # Иногда добавляем случайное решение из памяти для разнообразия
            if self.solution_memory and np.random.random() < 0.3:
                new_population[-1] = np.random.choice(list(self.solution_memory))

            population = new_population

            # Применяем улучшение к каждому решению в популяции
            for i in range(len(population)):
                population[i] = self.improve_response(
                    population[i],
                    evaluated_population[0]['feedback'],  # Используем критику лучшего решения
                    initial_task,
                    self.improvement_len,
                    adaptive_temp
                )

        return best_result['value']