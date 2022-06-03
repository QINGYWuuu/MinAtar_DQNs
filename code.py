#给定一个字符串，找出不含有重复字符的最长子串的长度。

#示例：

#给定 "abcabcbb" ，没有重复字符的最长子串是 "abc" ，那么长度就是3。

#给定 "bbbbb" ，最长的子串就是 "b" ，长度是1。

#给定 "pwwkew" ，最长子串是 "wke" ，长度是3。请注意答案必须是一个子串，"pwke" 是 子序列


def find_s(input_s):
    len_s = len(input_s)
    if len_s == 0:
        print(0)
    head = 0
    tail = 1

    max_s = input_s[head:tail]
    max_len = 1


    while tail < len_s:

        if input_s[tail] not in input_s[head:tail]:
            tail += 1
            if tail-head > max_len:
                max_s = input_s[head:tail]
                max_len = tail - head
        else:
            for id in range(head, tail):
                if input_s[id] == input_s[tail]:
                    head = id + 1
                    break

    print(max_s, max_len)


find_s("pwwkew")
