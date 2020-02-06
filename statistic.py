# -*-coding=utf-8-*-
import Levenshtein as Lev


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
    # print Lev.distance(s1, s2)
    print(Lev.distance(s1, s2))
    return Lev.distance(s1, s2)


def cer_s(data):
    count = 0
    size = 0
    # print('att_predict', data)
    for pair in data:
        index = Lev.distance(pair[1].strip(), pair[0].strip())
        print(pair[1], pair[0], index, len(pair[1].strip()), len(pair[0].strip()))
        count += Lev.distance(pair[1].strip(), pair[0].strip())
        size += len(pair[1].strip())
        # print(size)
    # print(size, count, float(count) / size)
    return count, float(count) / size


if __name__ == '__main__':
    #cer_s([("一些古钱币", "你懂什么"), ("我才这么了", "我也这么想"),
     #      ("你下么使了", "你凭什么指使我"), (" 你内哄", "起内哄"),
     #      ("你不啊", "死都不要干"), ("良中", "考虑中"), ("有我好电了", "我快饿扁了"),
      #     ("真再了", "就这样"), ("重人尊透帐", "原始人使用石器"), ("军多勇业", "蛊惑人心")])
    #cer("你不啊", "不啊")
    #cer("不啊", "你不啊啊")
    cer('我','我是在变快')
