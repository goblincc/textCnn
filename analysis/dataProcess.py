

def process():
    key_dict = {}
    nick2uid = {}
    with open("../data/tb_ks_keywd_reflect_keyword.txt", "rt", encoding='utf-8') as f:
        for line in f:
            s = line.split('\t')
            if s[0].strip() not in key_dict:
                key_dict[s[0].strip()] = s[1].strip()
    with open("../data/tb_ks_keywd_reflect_lettnum.txt", "rt", encoding='utf-8') as f:
        for line in f:
            s = line.split('\t')
            if s[0].strip() not in key_dict:
                key_dict[s[0].strip()] = s[1].strip()

    with open("../data/uid_nick.txt", "rt", encoding='utf-8') as f:
        for line in f:
            s = line.split('\t')
            nick = s[1]
            nick_word = []
            for i in nick:
                if i in key_dict:
                    i = key_dict[i]
                if i.isdigit():
                    nick_word.append(i)
            nick = "".join(nick_word)
            if nick not in nick2uid.keys():
                nick2uid.setdefault(nick, []).append(s[0].strip())
            else:
                nick2uid[nick].append(s[0].strip())
    j = 0
    for key, value in nick2uid.items():
        if len(value) >= 5 and len(key) == 10 and key != 100000:
            print(key, value)


if __name__ == '__main__':
    process()