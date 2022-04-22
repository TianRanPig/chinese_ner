import os
import tkinter as tk
from tkinter import ttk, messagebox

from torch_ner.source.ner_predict import get_entities_result


def output():
    for item in tv.get_children():
        tv.delete(item)
    model_path = os.path.join(os.path.abspath('.'), 'output/20220417160929')
    if len(e1.get()) == 0:
        messagebox.showinfo("提示","输入语句为空！")
        return
    result = get_entities_result(e1.get(), model_path)
    list = []
    for i, item in enumerate(result):
        data = (i, item["value"], item["type"], item["begin"], item["end"])
        list.append(data)
    if len(list) != 0:
        for data in list:
            tv.insert('', 'end', values=data)
    else:
        messagebox.showinfo("提示","没有识别出命名实体！")


def clear():
    var1.set("")
    # 清空列表
    for item in tv.get_children():
        tv.delete(item)


# 调用Tk()创建主窗口
window = tk.Tk()
# 给主窗口起一个名字，也就是窗口的名字
window.title('中文命名实体识别')
var1 = tk.StringVar()
# 设置窗口大小变量
width = 900
height = 500
# 窗口居中，获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
screenwidth = window.winfo_screenwidth()
screenheight = window.winfo_screenheight()
size_geo = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
window.geometry(size_geo)

window.grid_columnconfigure(0, weight=1, uniform='a')
window.grid_columnconfigure(1, weight=1, uniform='a')
window.grid_columnconfigure(2, weight=1, uniform='a')
window.grid_columnconfigure(3, weight=1, uniform='a')
window.grid_columnconfigure(4, weight=1, uniform='a')
window.grid_columnconfigure(5, weight=1, uniform='a')
window.grid_columnconfigure(6, weight=1, uniform='a')

tk.Label(window, text="请输入文本：", font=("宋体", 14)).grid(row=0, column=0, pady=20, sticky="e")

e1 = tk.Entry(window, font=("宋体", 14), textvariable=var1, width=70)
e1.grid(row=0, column=1, columnspan=6, pady=20, ipady=3)

# 开启主循环，让窗口处于显示状态
area = ('id', 'value', 'type', 'begin', 'end')
ac = ('i', 'v', 't', 'b', 'e')
tv = ttk.Treeview(window, columns=ac, show='headings', height=10)
for i in range(5):
    if i == 1:
        tv.column(ac[i], width=150, anchor='center')
    else:
        tv.column(ac[i], width=75, anchor='center')
    tv.heading(ac[i], text=area[i])
tv.grid(row=1, column=1, columnspan=5, pady=30)
tk.Button(window, text="识别", font=("宋体", 14), command=output, width=8).grid(row=2, column=2, pady=10)
tk.Button(window, text="清空", font=("宋体", 14), command=clear, width=8).grid(row=2, column=4, pady=10)
tips = "提示：共有10中实体标签，分别为：地址（address）、书名（book）、公司（company）、游戏（game）、政府"+'\n'+"（goverment）、电影（movie）、姓名（name）、组织机构（organization）、职位（position）、景点（scene）"
tk.Label(window, text=tips, fg='gray').grid(row=3,column=0,columnspan=7,pady=20)
window.mainloop()