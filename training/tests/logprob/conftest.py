import json

import pytest

from sft.logprob.targets import ByteVocabulary

BYTE_LEVEL = {}
_extra = 0
for _byte in range(256):
    if 33 <= _byte < 127 or 161 <= _byte < 173 or 174 <= _byte < 256:
        BYTE_LEVEL[_byte] = chr(_byte)
    else:
        BYTE_LEVEL[_byte] = chr(256 + _extra)
        _extra += 1


def byte_level(data):
    """ByteLevel BPE spelling of raw bytes, as tokenizer.json stores vocabulary keys."""
    return "".join(BYTE_LEVEL[b] for b in data)


def write_tokenizer(path, pieces, added=(), merges=()):
    """A ByteLevel tokenizer.json whose vocabulary is ``pieces`` (bytes) in id order.

    ``merges`` are (left, right) byte pairs in rank order.
    """
    vocab = {byte_level(piece): index for index, piece in enumerate(pieces)}
    if len(vocab) != len(pieces):
        raise ValueError("duplicate test vocabulary pieces")
    path.write_text(
        json.dumps(
            {
                "decoder": {"type": "ByteLevel"},
                "model": {
                    "vocab": vocab,
                    "merges": [
                        byte_level(left) + " " + byte_level(right)
                        for left, right in merges
                    ],
                },
                "added_tokens": [
                    {"id": len(pieces) + index, "content": content, "special": True}
                    for index, content in enumerate(added)
                ],
            }
        )
    )
    return ByteVocabulary(path)


@pytest.fixture
def vocab(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(
        json.dumps(
            {
                "decoder": {"type": "ByteLevel"},
                "model": {
                    "vocab": {"a": 0, "b": 1, "ab": 2, "Ġ": 3, "c": 4, "Ã": 5, "©": 6},
                    "merges": ["a b"],
                },
                "added_tokens": [{"id": 7, "content": "<think>", "special": True}],
            }
        )
    )
    return ByteVocabulary(path)


# Flash-side pieces for a completion with content and two tool calls, in id order.
MARKUP_PIECES = [
    b"a",  # 0
    b"b",  # 1
    b"ab",  # 2
    b" ",  # 3
    b"c",  # 4
    b"\n</think>\n\n",  # 5
    b"hi",  # 6
    b"!",  # 7
    b"\n\n<tool_call>\n<function=",  # 8
    b"python",  # 9
    b">\n<parameter=",  # 10
    b"code",  # 11
    b">\n",  # 12
    b"x",  # 13
    b"=",  # 14
    b"1",  # 15
    b"\n",  # 16
    b"f",  # 17
    b"()\n",  # 18  the teacher's "()" merges with Flash's newline
    b"</parameter>\n</function>\n</tool_call>",  # 19
    b"\n<tool_call>\n<function",  # 20
    b"=view",  # 21  the teacher's "view" has no boundary here
    b"_file",  # 22
    b"path",  # 23
    b"p",  # 24
    b"\n</parameter>\n<parameter=",  # 25
    b"view",  # 26
    b"_range",  # 27
    b"[",  # 28
    b", ",  # 29
    b"240",  # 30
    b"]",  # 31
    b"\n</parameter>\n",  # 32
    b"</function>\n</tool_call>",  # 33
    b"done",  # 34
    b"<tool_call>\n<function=",  # 35
]
MARKUP_ADDED = ("<|im_end|>", "<think>")  # ids 36, 37


@pytest.fixture
def markup_vocab(tmp_path):
    return write_tokenizer(tmp_path / "tokenizer.json", MARKUP_PIECES, MARKUP_ADDED)
