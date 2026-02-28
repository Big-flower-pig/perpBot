"""
PerpBot 装饰器模块

提供常用的装饰器：
- 重试机制
- 限流控制
- 执行时间记录
- 线程安全
- 参数验证
"""

import time
import threading
import functools
from typing import Callable, Any, Optional, Type, Tuple
from datetime import datetime
from collections import deque


def retry_on_failure(
    max_retries: int = 3,
    delay: float = 1.0,
    exponential_backoff: bool = True,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_failure: Optional[Callable] = None,
):
    """重试装饰器

    Args:
        max_retries: 最大重试次数
        delay: 初始延迟时间（秒）
        exponential_backoff: 是否使用指数退避
        exceptions: 需要重试的异常类型
        on_failure: 最终失败时的回调函数

    Example:
        @retry_on_failure(max_retries=3, delay=1)
        def unstable_api_call():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        # 记录重试
                        from utils.logger import get_logger
                        logger = get_logger("retry")
                        logger.warning(
                            f"函数 {func.__name__} 执行失败，"
                            f"第 {attempt + 1} 次重试，"
                            f"等待 {current_delay:.1f}秒",
                            extra={"data": {"error": str(e)}}
                        )
                        time.sleep(current_delay)
                        if exponential_backoff:
                            current_delay *= 2
                    else:
                        # 最终失败
                        if on_failure:
                            on_failure(e, *args, **kwargs)
                        raise last_exception

        return wrapper

    return decorator


def rate_limit(
    calls_per_second: float = 10,
    calls_per_minute: int = None,
    burst: int = None,
):
    """限流装饰器

    Args:
        calls_per_second: 每秒允许的调用次数
        calls_per_minute: 每分钟允许的调用次数
        burst: 允许的突发请求数

    Example:
        @rate_limit(calls_per_second=5)
        def api_call():
            ...
    """
    # 计算最小间隔
    min_interval = 1.0 / calls_per_second if calls_per_second else 0
    burst_limit = burst or int(calls_per_second)

    def decorator(func: Callable) -> Callable:
        # 线程安全的调用记录
        lock = threading.Lock()
        last_calls = deque(maxlen=max(burst_limit, calls_per_minute or 0))

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                now = time.time()

                # 移除过期的调用记录
                while last_calls and now - last_calls[0] > 1.0:
                    last_calls.popleft()

                # 检查每秒限制
                if len(last_calls) >= burst_limit:
                    sleep_time = 1.0 - (now - last_calls[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                        now = time.time()

                # 检查每分钟限制
                if calls_per_minute:
                    minute_calls = [
                        t for t in last_calls if now - t < 60
                    ]
                    if len(minute_calls) >= calls_per_minute:
                        sleep_time = 60 - (now - minute_calls[0])
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                            now = time.time()

                last_calls.append(now)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def log_execution_time(
    logger_name: str = "performance",
    threshold_ms: float = None,
):
    """执行时间记录装饰器

    Args:
        logger_name: 日志器名称
        threshold_ms: 阈值（毫秒），超过则记录警告

    Example:
        @log_execution_time(threshold_ms=1000)
        def slow_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end_time = time.perf_counter()
                duration_ms = (end_time - start_time) * 1000

                from utils.logger import get_logger
                logger = get_logger(logger_name)

                if threshold_ms and duration_ms > threshold_ms:
                    logger.warning(
                        f"函数 {func.__name__} 执行时间过长: {duration_ms:.2f}ms",
                        extra={"data": {"duration_ms": duration_ms, "threshold_ms": threshold_ms}}
                    )
                else:
                    logger.performance(
                        operation=func.__name__,
                        duration_ms=duration_ms,
                    )

        return wrapper

    return decorator


def thread_safe(lock: threading.Lock = None):
    """线程安全装饰器

    Args:
        lock: 自定义锁，如果不提供则创建新锁

    Example:
        _lock = threading.Lock()

        @thread_safe(_lock)
        def shared_resource_access():
            ...
    """
    _lock = lock or threading.Lock()

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _lock:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def validate_params(*validators, **kwvalidators):
    """参数验证装饰器

    Args:
        *validators: 位置参数验证器
        **kwvalidators: 关键字参数验证器

    Example:
        @validate_params(
            lambda x: x > 0 or "必须大于0",
            name=lambda n: len(n) > 0 or "名称不能为空"
        )
        def create_order(amount, name):
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 验证位置参数
            for i, validator in enumerate(validators):
                if i < len(args):
                    result = validator(args[i])
                    if isinstance(result, str):
                        raise ValueError(f"参数 {i}: {result}")

            # 验证关键字参数
            for key, validator in kwvalidators.items():
                if key in kwargs:
                    result = validator(kwargs[key])
                    if isinstance(result, str):
                        raise ValueError(f"参数 {key}: {result}")

            return func(*args, **kwargs)

        return wrapper

    return decorator


def cache_result(ttl: float = 60, max_size: int = 128):
    """结果缓存装饰器

    Args:
        ttl: 缓存过期时间（秒）
        max_size: 最大缓存数量

    Example:
        @cache_result(ttl=30)
        def expensive_computation(n):
            ...
    """

    def decorator(func: Callable) -> Callable:
        cache = {}
        cache_lock = threading.Lock()
        access_order = []

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = (args, tuple(sorted(kwargs.items())))

            with cache_lock:
                now = time.time()

                # 检查缓存
                if key in cache:
                    value, timestamp = cache[key]
                    if now - timestamp < ttl:
                        return value

                # 清理过期缓存
                expired_keys = [
                    k for k, (_, t) in cache.items()
                    if now - t >= ttl
                ]
                for k in expired_keys:
                    del cache[k]
                    if k in access_order:
                        access_order.remove(k)

                # LRU 淘汰
                while len(cache) >= max_size and access_order:
                    oldest = access_order.pop(0)
                    if oldest in cache:
                        del cache[oldest]

            # 执行函数
            result = func(*args, **kwargs)

            with cache_lock:
                cache[key] = (result, now)
                access_order.append(key)

            return result

        return wrapper

    return decorator


def deprecated(message: str = None, version: str = None):
    """弃用警告装饰器

    Args:
        message: 弃用消息
        version: 弃用版本

    Example:
        @deprecated("请使用 new_function() 替代", version="2.0")
        def old_function():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import warnings

            msg = f"函数 {func.__name__} 已弃用"
            if version:
                msg += f" (从版本 {version})"
            if message:
                msg += f": {message}"

            warnings.warn(msg, DeprecationWarning, stacklevel=2)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def singleton(cls):
    """单例模式装饰器

    Example:
        @singleton
        class Database:
            ...
    """
    instances = {}
    lock = threading.Lock()

    @functools.wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            with lock:
                if cls not in instances:
                    instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


def catch_exceptions(
    *exceptions: Type[Exception],
    default: Any = None,
    logger_name: str = "error",
):
    """异常捕获装饰器

    Args:
        *exceptions: 要捕获的异常类型
        default: 发生异常时的默认返回值
        logger_name: 日志器名称

    Example:
        @catch_exceptions(ValueError, TypeError, default=None)
        def parse_data(data):
            ...
    """
    exception_types = exceptions or (Exception,)

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                from utils.logger import get_logger
                logger = get_logger(logger_name)
                logger.error(
                    f"函数 {func.__name__} 发生异常: {e}",
                    exc_info=True,
                )
                return default

        return wrapper

    return decorator


def measure_memory(logger_name: str = "memory"):
    """内存测量装饰器

    Example:
        @measure_memory()
        def memory_intensive_task():
            ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                import tracemalloc
                tracemalloc.start()
            except ImportError:
                return func(*args, **kwargs)

            try:
                result = func(*args, **kwargs)
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()

                from utils.logger import get_logger
                logger = get_logger(logger_name)
                logger.debug(
                    f"函数 {func.__name__} 内存使用: "
                    f"当前 {current / 1024:.2f}KB, 峰值 {peak / 1024:.2f}KB"
                )

                return result
            finally:
                tracemalloc.stop()

        return wrapper

    return decorator