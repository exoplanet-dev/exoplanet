import tensorflow as tf


def FlagsForFile(filename, **kwargs):
    flags = [
        '-x', 'c++',
        '-std=c++11',
        '-stdlib=libc++',
        # '-Wall',
        # '-Wextra',
        # '-Werror'
        '-I.'
    ]
    flags += tf.sysconfig.get_compile_flags()
    return {
        'flags': flags
    }
