# -*- coding: UTF-8 -*-
import json as js
import matplotlib.pyplot as plt
import numpy as np
from math import log10
from sys import setrecursionlimit
from threading import Thread, stack_size

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class Graph:
    def __init__(self):
        self.vertices = {}

    def addVertex(self, key):
        newVertex = Vertex(key)
        self.vertices[key] = newVertex
        return newVertex

    def __contains__(self, n):
        return n in self.vertices

    def addEdge(self, f, t, cost=0):
        if f not in self.vertices:
            self.addVertex(f)
        if t not in self.vertices:
            self.addVertex(t)
        self.vertices[f].addNeighbor(self.vertices[t], cost)

    def __iter__(self):
        return iter(self.vertices.values())


class Vertex:
    def __init__(self, num):
        self.id = num
        self.connectedTo = {}
        self.color = 'white'
        self.dist = 0
        self.disc = 0

    def addNeighbor(self, nbr, weight=0):
        self.connectedTo[nbr] = weight


# 对数据进行预处理

flist = js.load(open("Film.json", 'r', encoding='utf-8'))
for film in flist:
    film["id"] = film["_id"]["$oid"]
    film["type"] = film["type"].split(",")
    film["actor"] = film["actor"].split(",")
    del film["_id"]

# 演员及其演过的电影名单 -> dict（以id呈现）
# '林更新': ['快手枪手快枪手', '长城 The Great Wall', '梦回鹿鼎记',
#  '三少爷的剑', '痞子英雄2：黎明升起 痞子英雄2：黎明再起', '西游伏妖篇',
#  '狄仁杰之神都龙王', '智取威虎山', '2018把乐带回家', '我是路人甲',
#  '大话天仙', '同桌的妳', '谁的青春不迷茫', '机器人争霸']
actors = {}
for film in flist:
    for actor in film["actor"]:
        if actor in actors:
            actors[actor].append(film["id"])
        else:
            actors[actor] = [film["id"]]

# 合作演出名单 -> dict
# ('张静初', '林更新'): {'我是路人甲', '快手枪手快枪手'}（以id呈现）
coactors = {}
for film in flist:
    for act1 in film["actor"]:
        for act2 in film["actor"]:
            if act1 != act2:
                coactors_film = (min(act1, act2), max(act1, act2))
                if coactors_film in coactors:
                    coactors[coactors_film].add(film["id"])
                else:
                    coactors[coactors_film] = {film["id"]}

# 电影评分信息 -> dict
# '快手枪手快枪手': 5.1（以id呈现）
film_score = {}
for film in flist:
    film_score[film["id"]] = film["star"]

# 获得电影类别信息 -> dict
# '快手枪手快枪手': {'喜剧': 1, '动作': 1}（以id呈现）
film_type = {}
for film in flist:
    temp = {}
    for types in film["type"]:
        temp[types] = 1
    film_type[film["id"]] = temp
# 数据预处理结束


def merge(dict1, dict2):
    """
    合并两个字典，将其类别数量相加
    """
    for key in dict2:
        if key in dict1:
            dict1[key] += dict2[key]
        else:
            dict1[key] = dict2[key]
    return dict1


def buildGraph():
    """
    创建一个图
    """
    G = Graph()
    for coactor in coactors:
        G.addEdge(coactor[0], coactor[1], 1)
        G.addEdge(coactor[1], coactor[0], 1)

    for actor in actors:
        if actor not in G:
            G.addVertex(actor)

    return G


def dfs(vertex):
    """
    递归求出连通分支的演员名单
    """
    vertex.color = 'black'

    connected = [i for i in vertex.connectedTo]
    connected_colors = [i.color for i in connected]

    if "white" in connected_colors:
        actor_list = []

        for item in connected:
            if item.color == "white":
                result = dfs(item)
                actor_list.extend(result)

        actor_list.append(vertex.id)

        return actor_list  # 演员名单

    return [vertex.id]


def bfs(actor_list, G):
    """
    给当前连通分支中的所有顶点测定一次距离，
    并取最大值的最大值即是该连通分支的直径
    """
    # G实际上是是G.vertices，这里为了方便迭代直接重命名为G
    diameters = []
    for actor in actor_list:

        for _ in actor_list:
            G[_].dist = -1  # 设置默认距离-1
            G[_].disc = False  # 设置默认为未探索

        _bfs(actor, G, actor_list)

        lst = [G[i].dist for i in actor_list]
        diameters.append(max(lst))

    return max(diameters)


def _bfs(v, G, actor_list):
    """
    给当前顶点测定一次距离
    """
    actor = G[v]
    actor.dist = 0
    actor.disc = True
    makeDist(actor)
    nextv = actor.connectedTo

    while 1:
        lst = []
        for i in nextv:
            if i.disc is True:
                continue  # 如果是已探索的顶点则不再探索

            makeDist(i)

            i.disc = True
            lst.extend(i.connectedTo)

        nextv = lst

        discs = [G[actor].disc for actor in actor_list]
        if False not in discs:
            break  # 如果所有顶点已被探索则退出循环


def makeDist(vertex):
    """
    给当前顶点所关联的节点确定距离
    """
    for nbr in vertex.connectedTo:
        if nbr.dist == -1:
            nbr.dist = vertex.dist + 1


def funnyTest(answer):
    """
    一些乱七八糟的测试
    """
    print(len(actors['']))
    # 合作之王
    co = []
    for coactor in coactors:
        co.append((coactor, len(coactors[coactor])))

    co.sort(reverse=True, key=lambda x: x[1])
    print(co[:30])

    # 好兄弟&好闺蜜
    bro = 0
    sis = 0
    for film in flist:
        a = film["actor"]
        if '休·弗拉瑟' in a and '大卫·苏切' in a and '菲利普·杰克森' in a:
            bro += 1
        elif '小原乃梨子' in a and '野村道子' in a and '立壁和也' in a and '肝付兼太' in a:
            sis += 1

    print('bro=%d, sis=%d' % (bro, sis))

    # 统计烂片之王和好片之王
    actor_star = []
    for actor in actors:
        if len(actors[actor]) >= 12:
            temp = [film_score[i] for i in actors[actor]]
            avg = np.mean(temp)
            avg = round(avg, 2)
            actor_star.append((actor, avg))

    actor_star.sort(reverse=True, key=lambda x: x[1])
    print(len(actor_star))
    print(actor_star[-1:-40:-1])

    nl = []
    for film in flist:
        if film["director"] == "克里斯托弗·诺兰":
            nl.append(film["star"])
    avg = np.mean(nl)
    avg = round(avg, 2)
    print(len(nl))
    print(avg)


def plot(answer):
    """
    用matplotlib画图
    """
    n = 20

    # 前20名的规模
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.title('前20名的规模')
    plt.xlabel('前20名')
    plt.ylabel('规模')
    X = np.arange(1, n + 1)
    Y0 = [len(x[0]) for x in answer[:20]]
    Y1 = list(map(log10, Y0))
    plt.bar(X, Y1, facecolor='#39C5BB', edgecolor='black')

    i = iter(range(20))
    for x, y in zip(X, Y1):
        plt.text(x, y, '%d' % Y0[next(i)], ha='center', va='bottom')

    plt.xlim(0.5, n + 0.5)
    plt.xticks(list(range(1, n + 1)))
    plt.yticks([1, 2, 3, 4, 5],
               [r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$'])

    # 前20名的直径
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    plt.title('前20名的直径')
    plt.xlabel('前20名')
    plt.ylabel('直径')
    X = np.arange(1, n + 1)
    Y1 = [x[2] for x in answer[:20]]
    plt.bar(X, Y1, facecolor='#66CCFF', edgecolor='black')

    i = iter(range(20))
    for x, y in zip(X, Y1):
        plt.text(x, y, '%d' % Y1[next(i)], ha='center', va='bottom')

    plt.xlim(0.5, n + 0.5)
    plt.xticks(list(range(1, n + 1)))
    plt.yticks(list(range(-2, 6)))

    # 前20名的星级
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.title('前20名的平均星级')
    plt.xlabel('前20名')
    plt.ylabel('平均星级')
    X = np.arange(1, n + 1)
    Y1 = [x[3] for x in answer[:20]]
    plt.bar(X, Y1, facecolor='pink', edgecolor='black')

    i = iter(range(20))
    for x, y in zip(X, Y1):
        plt.text(x, y, '%.2f' % Y1[next(i)], ha='center', va='bottom')

    plt.xlim(0.5, n + 0.5)
    plt.xticks(list(range(1, n + 1)))
    plt.yticks(list(range(1, 11)))

    # 后20名的规模
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.title('后20名的规模')
    plt.xlabel('后20名（倒数）')
    plt.ylabel('规模')
    X = np.arange(1, n + 1)
    Y1 = [len(x[0]) for x in answer[-1:-21:-1]]
    plt.bar(X, Y1, facecolor='#39C5BB', edgecolor='black')

    i = iter(range(1, 21))
    for x, y in zip(X, Y1):
        plt.text(x, y, '%d' % Y1[-next(i)], ha='center', va='bottom')

    plt.xlim(0.5, n + 0.5)
    plt.xticks(list(range(1, n + 1)))
    plt.yticks([1, 2])

    # 后20名的直径
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.title('后20名的直径')
    plt.xlabel('后20名（倒数）')
    plt.ylabel('直径')
    X = np.arange(1, n + 1)
    Y1 = [x[2] for x in answer[-1:-21:-1]]
    plt.bar(X, Y1, facecolor='#66CCFF', edgecolor='black')

    i = iter(range(20))
    for x, y in zip(X, Y1):
        plt.text(x, y, '%d' % Y1[next(i)], ha='center', va='bottom')

    plt.xlim(0.5, n + 0.5)
    plt.xticks(list(range(1, n + 1)))
    plt.yticks([0, 1])

    # 后20名的星级
    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    plt.title('后20名的平均星级')
    plt.xlabel('后20名（倒数）')
    plt.ylabel('平均星级')
    X = np.arange(1, n + 1)
    Y1 = [x[3] for x in answer[-1:-21:-1]]
    plt.bar(X, Y1, facecolor='pink', edgecolor='black')

    i = iter(range(20))
    for x, y in zip(X, Y1):
        plt.text(x, y, '%.2f' % Y1[next(i)], ha='center', va='bottom')

    plt.xlim(0.5, n + 0.5)
    plt.xticks(list(range(1, n + 1)))
    plt.yticks(list(range(1, 11)))

    plt.show()
    return


def writeResult(answer, G):
    """
    将结果写进result.txt
    """
    f = open("result.txt", 'r+', encoding='utf-8')

    str1 = '=' * 16 + "问题一&问题二" + '=' * 16 + '\n'
    f.write(str1)

    str1 = '一共有%d个连通分支\n' % len(answer) + '-' * 45 + '\n'
    f.write(str1)

    for i in range(20):
        str2 = f"第{i+1}个连通分支：\n演员数量为：{len(answer[i][0])}，"
        str2 += f"直径长度为：{str(answer[i][2])}，"
        str2 += "排名前三（如果存在）的类别是："
        f.write(str2)

        if len(answer[i][1]) >= 3:
            str1 = str(answer[i][1][0][0]) + ' ' + str(
                answer[i][1][1][0]) + ' ' + str(answer[i][1][2][0]) + '，'

        else:
            str1 = ''
            for item in answer[i][1]:
                str1 += str(item[0]) + ' '
            str1 += '，'

        str1 += "平均星级为：" + str(answer[i][3]) + '\n'
        f.write(str1)

    for i in range(1, 21):
        str2 = f"倒数第{i}个连通分支：\n演员数量为：{len(answer[-i][0])}，"
        str2 += f"直径长度为：{str(answer[-i][2])}，"
        str2 += "排名前三（如果存在）的类别是："
        f.write(str2)

        if len(answer[-i][1]) >= 3:
            str1 = str(answer[-i][1][0][0]) + ' ' + str(
                answer[-i][1][1][0]) + ' ' + str(answer[-i][1][2][0]) + '，'

        else:
            str1 = ''
            for item in answer[-i][1]:
                str1 += str(item[0]) + ' '
            str1 += '，'

        str1 += "平均星级为：" + str(answer[i][3]) + '\n'
        f.write(str1)

    str1 = '=' * 18 + "周星驰" + '=' * 18 + '\n'

    films = {}
    co = G.vertices["周星驰"].connectedTo
    for cooperator in co:
        for film in actors[cooperator.id]:
            if film not in films:
                films[film] = film_score[film]

    types = {}
    for film in films:
        types = merge(types, film_type[film])

    avg = np.mean(list(films.values()))
    avg = round(avg, 2)

    typecount = [(j, types[j]) for j in types]
    typecount.sort(key=lambda x: x[1], reverse=True)

    starlist = [film_score[i] for i in actors["周星驰"]]
    avgstar = np.mean(starlist)

    str1 += f"周星驰出演电影的平均星级为{round(avgstar,2)}\n"
    str1 += f"周星驰和他的共同出演者总共有{len(co)+1}人\n"
    str1 += f"他们各自一共出演了{len(films)}部电影\n"
    str1 += f"所出演的电影平均星级为{avg}\n"
    str1 += f"电影所属类别的前三名：{typecount[0]} {typecount[1]} {typecount[2]}"

    f.write(str1)
    f.close()


def main():
    """
    主函数，整个程序的入口
    """
    G = buildGraph()
    print('Graph done!')
    answer = []
    for actor in G:
        if actor.color == "white":
            result = [dfs(actor)]
            # result[0]->该连通分支的演员名单
            # result[1]->该连通分支的分类统计
            # result[2]->该连通分支的直径
            # result[3]->该连通分支的平均星级

            # 第一题
            f = set()
            for player in result[0]:
                f.update(actors[player])

            dict0 = {}
            for i in f:
                dict0 = merge(dict0, film_type[i])

            tem = [(j, dict0[j]) for j in dict0]
            tem.sort(key=lambda x: x[1], reverse=True)
            result.append(tem)

            # 第二题
            if len(result[0]) < 100:
                diameter = bfs(result[0], G.vertices)
                result.append(diameter)
            else:
                # 最大的那个不计算直径
                result.append(-1)

            # 求出平均星级
            star = {}
            for shit in result[0]:
                lst = actors[shit]
                for film in lst:
                    if film not in star:
                        star[film] = film_score[film]
            avg = np.mean(list(star.values()))
            avg = round(avg, 2)
            result.append(avg)

            answer.append(result)

    answer.sort(reverse=True, key=lambda x: len(x[0]))

    writeResult(answer, G)
    funnyTest(answer)
    plot(answer)
    return 0


setrecursionlimit(10**6)
stack_size(2**27)
t = Thread(target=main)
t.start()
t.join()
