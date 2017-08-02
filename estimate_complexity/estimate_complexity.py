import numpy as np
import timeit
import logging
import pynverse
from scipy.misc import derivative
import matplotlib.pyplot as plt
import scipy.optimize as scp
from timeout_decorator import timeout_decorator


# functions that we're going to try to fit
def n_log_n(x, a, b, c):
    return a * x * np.log2(x) + b * x + c


def log_n(x, a, b):
    return a * np.log2(x) + b


def linear(x, a, b):
    return a * x + b


def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c


def exponential(x, a, b, c):
    return a * 2 ** x + b * x + c


def square_root(x, a, b):
    return a * np.sqrt(x) + b


def avg(l):
    return sum(l) / len(l)


# colors used when drawing several plots at once
colors = ('#00FFFF', '#FF00FF', '#FFFF00', '#0000FF', '#00FF00')


# function to calculate square error
def square_error(f, values, times):
    sq_err = 0
    for i in range(len(values)):
        sq_err += (f(values[i]) - times[i]) ** 2
    return sq_err


# generic exception for my class
class ComplexityException(Exception):
    pass


# function that calculates the time for a given n
def _get_time(to_run, setup, n, mutability):
    if mutability:
        return avg(
            timeit.repeat('fun(*my_args, **my_kwargs)',
                          'my_args, my_kwargs = setup({})'.
                          format(n), repeat=5,
                          globals={'fun': to_run,
                                   'setup': setup},
                          number=1))
    else:
        return min(timeit.repeat('fun(*my_args, **my_kwargs)',
                                 'my_args, my_kwargs = setup({})'.
                                 format(n), repeat=2,
                                 globals={'fun': to_run, 'setup': setup},
                                 number=5)) / 5


# function that calculates the times for various n values
# times and values are mutable lists so even though an exception
# might arise we do not lose the information
# several arguments to tweak the parameters of the finding time taken for n
def _find_times(setup, to_run, mutability, times, values,
                begin_val, upper_bound, n_increment, timeout):
    n = begin_val
    while 1:
        if n > upper_bound:
            logging.log(logging.INFO, 'N reached the upper bound of {}'.
                        format(upper_bound))
            break
        try:
            time = timeout_decorator.timeout(int(timeout / 4))(
                lambda: _get_time(to_run, setup, n, mutability))()
            times.append(time)
            values.append(n)
            logging.log(logging.DEBUG, 'N = {} took {} time'.format(n, time))
            n = n_increment(n, time)
        except ComplexityException:
            logging.log(logging.DEBUG, 'time for N = {} '
                                       'exceeded the soft limit'.format(n))
            break
    return values, times


# exception for when functions throw an exception
class FunctionError(Exception):
    pass


def fill_coefs(fun, *args):
    return lambda x: fun(x, *args)


# function that takes as input values and times taken for them
# and returns functions that are fitted to these values
def _fit_functions(times, values):
    if len(values) != len(times):
        times.pop()
    mean = np.mean(values, axis=0)
    variance = np.var(values / mean, axis=0)
    n = len(values)
    # if variance is so small there is no way of telling
    # what the complexity is since this is
    if variance < 1e-10:
        logging.log(logging.INFO, 'Probably constant, very small variance')
        return mean
    if n < 6:
        if variance < 1e-8:
            logging.log(logging.INFO, 'Probably constant, small variance')
            return mean
        elif n > 3:
            logging.log(logging.INFO, 'Possibly exponential '
                                      'time complexity, might be even worse')
            something, _ = scp.curve_fit(exponential, values, times)
            return (lambda x: exponential(x, *something), 'exponential'),
        else:
            logging.log(logging.WARNING, 'Not enough points '
                                         'to approximate the complexity')
            return None

    vals_for_nlogn, _ = scp.curve_fit(n_log_n, values, times)
    vals_for_linear, _ = scp.curve_fit(linear, values, times)
    vals_for_logn, _ = scp.curve_fit(log_n, values, times)
    vals_for_quad, _ = scp.curve_fit(quadratic, values, times)
    vals_for_sqrt, _ = scp.curve_fit(square_root, values, times)

    nlognlambda = fill_coefs(n_log_n, *(vals_for_nlogn))
    linearlambda = fill_coefs(linear, *(vals_for_linear))
    lognlambda = fill_coefs(log_n, *(vals_for_logn))
    quadl = fill_coefs(quadratic, *(vals_for_quad))
    sqrtl = fill_coefs(square_root, *(vals_for_sqrt))

    funs = ((nlognlambda, 'nlog(n)'), (linearlambda, 'n'),
            (lognlambda, 'log(n)'), (quadl, 'n^2'), (sqrtl, 'sqrt(n)'))
    logging.log(logging.DEBUG, 'Finished fitting functions')
    return funs


def clean_up(f):
    logging.log(logging.DEBUG, 'Calling the clean up function')
    try:
        f
    except Exception as e:
        raise FunctionError('Call to the clean up function failed', e.args)


# exception used in functions n <-> time
class OutOfDomainException(Exception):
    pass


def safe_domain_and_image(f):
    def safe_f(*args):
        try:
            return f(*args)
        except Exception as e:
            logging.log(logging.WARNING, 'exception when calling {}'.
                        format(f.__name__))
            raise OutOfDomainException('The function {} was '
                                       'called with invalid arguments'
                                       '(could be too large '
                                       'or too small)', e.args)

    return safe_f


# the main class
class Complexity:
    def __init__(self, function_to_time, setup=lambda n: ([], {}),
                 one_time_setup_and_cleanup=(None, None),
                 timeout: int = 30,
                 begin_with_n=1, upper_limit_on_n=np.inf,
                 n_increment=lambda n, time: n * 2
                 if n < 10000 else int(n * 1.3),
                 mutable: bool = False, debug_log: bool = False):
        """
        DO NOT MODIFY THE PARAMETERS THAT HAVE TO DO WITH N
        UNLESS YOU KNOW WHAT YOU'RE DOING

        :type upper_limit_on_n: if n reaches this
        limit we don't calculate it any further

        :type begin_with_n: iterations will start with the given n

        :type timeout: determines what's the maximum amount of
        time we can spend calculating running the function

        :type debug_log: if set to True debugging information
        will be written to wherever the logger is configured

        :type n_increment: function which takes previous n and the time it
        took to evaluate function at n as arguments and returns the next n

        :type mutable: determines whether the setup has
        to be rerun before every iteration of the function_to_time

        :type one_time_setup_and_cleanup: a tuple of functions
        that will be called only once
        before and after calculating complexity

        :type setup: function that returns a tuple
        of positional and keyword arguments that are
        to be used when timing the function
        the arguments have to agree with the arguments
        that function_to_time takes
        it will be called every time n changes
        """
        if debug_log:
            logging.basicConfig(level='DEBUG')

        logging.log(logging.INFO, 'calculating complexity of {}'.
                    format(function_to_time.__name__))

        self.timeout = timeout
        self.function = function_to_time
        self.setup = setup
        self.initialize = one_time_setup_and_cleanup[0]
        self.close_resources = one_time_setup_and_cleanup[1]

        """
        Here's where the magic happens
        """
        try:
            if self.initialize:
                self.initialize()
        except Exception as e:
            raise FunctionError("One time setup failed", e.args)
        times = []
        values = []
        try:
            timeout_decorator.timeout(timeout)(
                lambda: _find_times(self.setup,
                                    self.function, mutable, times,
                                    values,
                                    begin_with_n, upper_limit_on_n,
                                    n_increment,
                                    timeout))()

        except timeout_decorator.TimeoutError:
            logging.log(logging.DEBUG, 'Finished running '
                                       'iterations with different N\'s')

        self.times = times
        self.values = values

        functions = _fit_functions(times, values)
        if not functions:
            if one_time_setup_and_cleanup[1]:
                clean_up(one_time_setup_and_cleanup[1])
            raise ComplexityException("Couldn't determine "
                                      "the complexity\n"
                                      "Possible causes:\n"
                                      "Huge constants\n"
                                      "Too little time given to "
                                      "approximate complexity\n"
                                      "Hyperexponential complexity")

        if not isinstance(functions, tuple):
            self.minimum = (-1, (lambda x: functions, 'constant'))
            logging.log(logging.INFO, 'The function {} most likely '
                                      'has constant time complexity O(1)'.
                        format(function_to_time.__name__))
            self.functions = (self.minimum[1],)
            return

        self.functions = functions
        square_errors = []
        for f in functions:
            if derivative(f[0], x0=2 * values[-1], dx=1e-5) > 0:
                logging.log(logging.DEBUG, 'calculating square error for {}'.
                            format(f[1]))
                square_errors.append((square_error(f[0], values, times), f))
                logging.log(logging.DEBUG, '{} has square error of {}'.
                            format(f[1], square_errors[-1]))

        if not square_errors:
            if one_time_setup_and_cleanup[1]:
                clean_up(one_time_setup_and_cleanup[1])
                return
            else:
                raise ComplexityException('None of the functions could '
                                          'fit the complexity')

        minimum = (np.inf, None)
        for elem in square_errors:
            if elem[0] < minimum[0]:
                minimum = elem

        logging.log(logging.INFO, 'The complexity of {} is most likely {}'.
                    format(function_to_time.__name__, minimum[1][1]))

        self.minimum = minimum

    def get_time_to_elements_fun(self):
        try:
            inverse = pynverse.inversefunc(self.minimum[1][0],
                                           domain=[1, int(1e12)])
            return safe_domain_and_image(inverse)
        except:
            raise ComplexityException("The approximation "
                                      "couldn't be inverted")

    def get_elements_to_time_fun(self):
        return safe_domain_and_image(self.minimum[1][0])

    def get_complexity(self) -> str:
        return self.minimum[1][1]

    def draw_plot_with_all(self):
        input_vector = np.linspace(1, self.values[-1] * 1.2, 10000)
        for i, f in enumerate(self.functions):
            plt.plot(input_vector, f[0](input_vector), color=colors[i])
        plt.plot(self.values, self.times, 'o', color='#FF0000')
        plt.show()

    def draw_plot_with_chosen(self):
        input_vector = np.linspace(1, self.values[-1] * 1.2, 10000)
        plt.plot(input_vector,
                 self.minimum[1][0](input_vector), color='#0000FF')
        plt.plot(self.values, self.times, 'o', color='#FF0000')
        plt.show()
