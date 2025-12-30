"""
Math utility functions for formula evaluation
"""
import math
import re

def evaluate_formula(formula: str, a=0, b=0, c=0, d=0) -> float:
    """
    计算字符串数学公式
    
    支持的运算符和函数：
    - 基本运算：+, -, *, /, //, %, **
    - 比较运算：>, <, >=, <=, ==, !=
    - 数学函数：abs, pow, round, ceil, floor, sqrt, exp, log, log10
    - 三角函数：sin, cos, tan, asin, acos, atan
    - 常量：pi, e
    
    Args:
        formula: 数学公式字符串，可以使用变量a、b、c、d
        a: 变量a的值
        b: 变量b的值
        c: 变量c的值
        d: 变量d的值
    
    Returns:
        计算结果
    
    Examples:
        >>> evaluate_formula("a + b", 1, 2)
        3.0
        >>> evaluate_formula("pow(a, 2)", 5)
        25.0
        >>> evaluate_formula("ceil(a / b)", 5, 2)
        3.0
        >>> evaluate_formula("(a>b)*b+(a<=b)*a", 5, 3)
        3.0
        >>> evaluate_formula("(a>b)*b+(a<=b)*a", 2, 3)
        2.0
    """
    # 安全的数学函数白名单
    safe_dict = {
        # 基本运算
        'abs': abs,
        'pow': pow,
        'round': round,
        # 数学函数
        'ceil': math.ceil,
        'floor': math.floor,
        'sqrt': math.sqrt,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        # 三角函数
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        # 常量
        'pi': math.pi,
        'e': math.e,
        # 变量
        'a': float(a),
        'b': float(b),
        'c': float(c),
        'd': float(d),
    }
    
    try:
        # 使用eval计算公式，限制可用的函数和变量
        result = eval(formula, {"__builtins__": {}}, safe_dict)
        return float(result)
    except Exception as e:
        raise ValueError(f"公式计算错误: {str(e)}")


def ceil_value(value: float) -> int:
    """向上取整"""
    return math.ceil(value)


def floor_value(value: float) -> int:
    """向下取整"""
    return math.floor(value)


def round_value(value: float, decimals: int = 0) -> float:
    """
    四舍五入
    
    Args:
        value: 要取整的值
        decimals: 保留小数位数
    
    Returns:
        四舍五入后的值
    """
    return round(value, decimals)


def power(base: float, exponent: float) -> float:
    """计算幂运算"""
    return math.pow(base, exponent)


def sqrt_value(value: float) -> float:
    """计算平方根"""
    if value < 0:
        raise ValueError("不能对负数求平方根")
    return math.sqrt(value)


def add(a: float, b: float) -> float:
    """加法"""
    return a + b


def subtract(a: float, b: float) -> float:
    """减法"""
    return a - b


def multiply(a: float, b: float) -> float:
    """乘法"""
    return a * b


def divide(a: float, b: float) -> float:
    """除法"""
    if b == 0:
        raise ValueError("除数不能为零")
    return a / b
