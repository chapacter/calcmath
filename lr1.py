import numpy as np
import sympy as sp
import scipy as scp
import math
import matplotlib.pyplot as plt

x, y, z = sp.symbols("x, y, z")

# Вариант 3

def task1(a1 = 5, a2 = 5, a3 = 20): # 1. Создать 5х5 и ед. 20х20
    print("\nЗадание 1")
    A_of_ones = np.ones((a1, a2), dtype=int) # матрица 5x5 из единиц.
    print("\nматрица 5x5 из единиц:\n", A_of_ones)

    A_eye = np.eye(a3)  # Единичная матрица 20х20
    print("\nЕдиничная матрица 20х20:\n", A_eye)

def task2(): # 2. Определитель матрицы
    print("\nЗадание 2")
    A = np.array([[3, 1, 2, 3, 2], [1, 2, 3, 3, 4], [2, 3, 4, 2, 1], [3, 0, 0, 5, 0], [2, 0, 0, 4, 0]], dtype=int)
    print("\nматрица:\n", A, " \nОпределитель матрицы: ", int(np.linalg.det(A)))

def task3(x_val = -2.01, y_val = math.sqrt(5), expr = (2/x**2 + 3/y**2)*(x+3*y)): # Упростите выражение.
    print("\nЗадание 3")
    s_expr = sp.simplify(expr)
    print(f"Исходное:    {expr} \nУпрощенное:  {s_expr}"
          f" \n\nРезультат: {s_expr.subs({x: x_val, y:y_val})} \nпри: x = {x_val}, y = {y_val} \n")
    return s_expr

def task4(expr): #Найти частные производные от выражения из задания выше
    print("\nЗадание 4")
    x_diff = sp.diff(expr, x)
    s_x_diff = sp.simplify(x_diff)

    y_diff = sp.diff(expr, y)
    s_y_diff = sp.simplify(y_diff)

    print(f"Выражение: {expr}\n"
          f"Производная по х:   {x_diff}   =>   {s_x_diff}\n"
          f"            по y:   {y_diff}   =>   {s_y_diff}")

def task5(a1 = 4, a2 = 4, am = 0, aM = 5): #  Cоздать случайную матрицу A из целых чисел [0,5] размера 4x4. Создать вектор-столбец B подходящего размера. Решить систему AX = B.
    print("\nЗадание 5")
    A = np.random.randint(am,aM+1,(a1, a2))
    B = np.random.randint(am, aM+1, (a1, 1))
    print(f"Матрица A: \n{A} \n\nМатрица B: \n{B} \n")

    try:
        X = np.linalg.solve(A, B)
        print("\nРешение X:")
        print(X)
        # Проверка
        print("\nПроверка (A * X): \n", np.dot(A, X))

    except np.linalg.LinAlgError:
        print("\nМатрица A вырождена, система не имеет единственного решения.")

        X, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        print("\nРешение методом наименьших квадратов (одно из возможных):", X)
        print("\nНевязка ||A X - B||^2:", residuals[0] if residuals.size > 0 else 0)

def task6(am = 0, aM = 1/3, expr = sp.cosh(3*x)**2): # Вычислить интеграл через scp и sp
    print("\nЗадание 6")
    print(f"Интеграл[{am},{aM}]: {expr}")

    #SymPy
    integral_sp = sp.integrate(expr, (x, am, aM))
    print("Результат (SymPy): ", integral_sp)

    # SciPy
    def expr1(x):
        return np.cosh(3 * x) ** 2
    result_scp, error_est = scp.integrate.quad(expr1, am, aM)
    print("Результат (SciPy): ", result_scp)

def task7(am = 0, aM = 1, am1 = 0, aM1 = 1-x, am2 = 0, aM2 = 1-x-y, expr = x+y+z): # Вычислить интеграл 2мя способами
    print("\nЗадание 7")
    expr = sp.sympify(expr)

    integral_sp = sp.integrate(expr, (z, am2, aM2), (y, am1, aM1), (x, am, aM))
    print(f"∫[{am},{aM}] ∫[{am1},{aM1}] ∫[{am2},{aM2}]: ({expr})dx dy dz \nРезультат (sympy): ", integral_sp,
          "\nЧисленно:         ", integral_sp.evalf())

    f_scipy = sp.lambdify((z, y, x), expr, modules='numpy')
    y_lower = sp.lambdify(x, am1, modules='numpy')
    y_upper = sp.lambdify(x, aM1, modules='numpy')
    z_lower = sp.lambdify((x, y), am2, modules='numpy')
    z_upper = sp.lambdify((x, y), aM2, modules='numpy')

    integral_scp, error_est = scp.integrate.tplquad(f_scipy, am, aM, y_lower, y_upper, z_lower, z_upper)
    print("Результат (sciPy):", integral_scp,
          "\nПогрешность:      ", error_est)

def task8(): # Построить графики функций
    print("\nЗадание 8\nПамагите...")

    plt.figure(figsize=(10, 5), dpi=80)
    ax = plt.subplot(111)

    # Границы
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    def f1(x):
        return 2 * np.cos(x - np.pi / 4)
    def f2(x):
        return x + 3

    # Подготовить данные, использовать распаковку последовательности
    X = np.linspace(-2*np.pi, 2*np.pi, 256, endpoint=True)
    C, L = f1(X), f2(X)


    plt.plot(X, C, color="blue", linewidth=2.5, linestyle="-", label="Cos Func")
    plt.plot(X, L, color="red", linewidth=2.5, linestyle="--", label="Line Func")


    plt.xlim(X.min() * 1.1, X.max() * 1.1)

    # Изменить метку на оси координат
    plt.xticks([-2*np.pi, -3*np.pi / 2, -np.pi, -np.pi / 2, 0, np.pi / 2, np.pi, 3*np.pi / 2, 2*np.pi ], [r'$-2\pi$', r'$-3\pi/2$', r'$-\pi$', r'$-\pi/2$', r'$0$', r'$+\pi/2$', r'$+\pi$', r'$+3 \pi/2$', r'$+ 2\pi$' ])
    plt.yticks([-2, -1, 1, 2], [r'$-2$', r'$-1$', r'$+1$', r'$+2$'])

    plt.ylim(C.min() * 1.1, C.max() * 1.1)
    # plt.yticks([-2, -1, +1, +2], [r'$-2$', r'$-1$', r'$+1$', r'$+2$'])
    #добавляем точку
    # ax.scatter(x=105, y=110, c='g')



    # Стрелки для осей
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    ax.annotate('', xy=(x_max * 0.95, 0), xytext=(x_max * 0.9, 0), arrowprops=dict(arrowstyle='->', color='black', lw=2))  # Стрелка в конце оси X
    ax.annotate('', xy=(0, y_max * 0.95), xytext=(0, y_max * 0.9), arrowprops=dict(arrowstyle='->', color='black', lw=2))  # Стрелка для оси Y (вверх)

    # Подписи осей
    plt.xlabel('    x', labelpad=10) # Вариант 1
    plt.ylabel('    y')
    ax.text(x_max, 0, 'x', ha='left', va='center', fontsize=12) # Вариант 2
    ax.text(0, y_max, 'y', ha='center', va='bottom', fontsize=12)

    # Поиск точки пересечения
    def diff(x):
        return f1(x) - f2(x)
    x_cross = scp.optimize.fsolve(diff, -3.5)[0]  # начальное приближение выбрано по графику
    y_cross = x_cross + 3
    # print(f"Точка пересечения: x = {x_cross:.4f}, y = {y_cross:.4f}")
    ax.scatter([x_cross], [y_cross], color='orange', s=100, zorder=5, label='Точка Пересечения')

    plt.legend(loc='upper right', frameon=False)
    plt.grid()
    plt.show()
# - - - - - - - - - - - - - - - - - - - -

# task1()
# task2()
# expr = task3()
# task4(expr)
# task5()
# task6()
# task7()
task8()