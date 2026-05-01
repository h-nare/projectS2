import ctypes

lib = ctypes.CDLL("./tfidf.so")

MAX_WORDS = 200
MAX_LEN = 50


class Word(ctypes.Structure):
    _fields_ = [
        ("word", ctypes.c_char * MAX_LEN),
        ("tf", ctypes.c_int)
    ]


def compute_tf(text):
    words_array = (Word * MAX_WORDS)()

    text_bytes = ctypes.create_string_buffer(text.encode("utf-8"))

    lib.compute_tf.argtypes = [ctypes.c_char_p, ctypes.POINTER(Word)]
    lib.compute_tf.restype = ctypes.c_int

    size = lib.compute_tf(text_bytes, words_array)

    result = {}

    for i in range(size):
        word = words_array[i].word.decode("utf-8", errors="ignore")
        result[word] = words_array[i].tf

    return result