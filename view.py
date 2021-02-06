from tkinter import filedialog
from tkinter import ttk
import csv
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import matplotlib
from tkinter import *
import os, time

#def round5(number):
#    return round(number, 5)
class View:
    def __init__(self):
        super().__init__()
        self.logic = None

    def start(self):
        self.root = Tk()
        self.root.title("Хроматографический анализ трансформаторного масла")
        self.root.geometry("810x690")
        self.root.resizable(width=False, height=False)
        self.root.grid_rowconfigure(5)
        self.root.grid_columnconfigure(5)

        matplotlib.use('TkAgg')

        #self.graphMat()


        # все элементы GUI
        # создание меню приложения
        self.menu = Menu(self.root)
        self.menu.add_command(label="Открыть", command=self.openCSV)
        self.menu.add_command(label="Выйти", command=self.root.destroy)
        self.root.config(menu=self.menu)

        # создание текстового поля "Тип трансформатора"
        Label(text = 'Тип трансформатора:', font=("Segoe UI", '10')).grid(row = 0, column = 0)

        # создание комбобокса "Тип трансформатора"
        self.trans = ttk.Combobox(self.root, values = ['35 кВ', '220 кВ','500 кВ'], font=("Segoe UI", '10'))
        self.trans.grid(row = 0, column = 1)
        self.trans.current(1)

        # создание текстового поля "Срок эксплуатации"
        Label(text = 'Срок эксплуатации:',font=("Segoe UI", '10')).grid(row = 0, column = 2)

        # создание комбобокса
        self.vars = IntVar()
        self.vars.set(0)
        Radiobutton(text = 'до 5 лет', font=("Segoe UI", '10'), variable = self.vars, value = 0).grid(row=0, column=3)
        Radiobutton(text='более 5 лет', font=("Segoe UI", '10'), variable = self.vars, value = 1).grid(row=0, column=4)


        # создание treeview - таблицы со значениями газов
        columns = ("#1", "#2", "#3", "#4", "#5")
        self.tree = ttk.Treeview(self.root, show="headings", columns=columns)
        self.tree.column("#1", minwidth=0, width=20, stretch=True)
        self.tree.column("#2", minwidth=0, width=150, stretch=True)
        self.tree.column("#3", minwidth=0, width=150, stretch=True)
        self.tree.column("#4", minwidth=0, width=150, stretch=True)
        self.tree.column("#5", minwidth=0, width=150, stretch=True)
        self.tree.heading("#1", text='№', anchor= CENTER)
        self.tree.heading("#2", text='H2', anchor= CENTER)
        self.tree.heading("#3", text='CO', anchor= CENTER)
        self.tree.heading("#4", text='C2H4', anchor= CENTER)
        self.tree.heading("#5", text='C2H2', anchor= CENTER)
        self.tree.tag_configure('gray', background='#cccccc')
        ysb = Scrollbar(self.root, orient = VERTICAL, command = self.tree.yview)
        self.tree.configure(yscroll = ysb.set)
        self.tree.grid(row=1, column=0, columnspan=5, sticky=N+S+W+E)
        ysb.grid(row=1, column=5, sticky = N + S)

        self.tree.bind("<<TreeviewSelect>>", self.print_selection)

        # вывод сообщения о результатах прогнозирования
        self.predict_label = Label(text='', font=("Segoe UI", '10'))
        self.predict_label.grid(row=3, column=0, columnspan=6, sticky=S + W)
        self.time_forecast = Label(text='', font=("Segoe UI", '10'))
        self.time_forecast.grid(row=4, column=0, columnspan=6, sticky=S + W)

        # графики
        self.gas = [0, 1, 2, 3]
        self.fig = plt.figure(figsize=(7.2, 3), dpi=110, constrained_layout=True)
        canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        canvas.get_tk_widget().grid(row=2, column=0,  columnspan=5, pady=10, sticky=N+S+W+E)
        plt.rcParams['font.family'] = "Segoe UI"
        plt.rcParams['font.size'] = "8"
        # график ПДЗ, ДЗ и тд
        plt.title("Хроматографический анализ трансформаторного масла")
        plt.ylabel('Объемная концентрация, в долях')
        plt.xticks(range(4), ['H2', 'CO', 'C2H4', 'C2H2'])
        plt.grid()
        self.fig.canvas.draw()


        # отображение окна
        self.root.mainloop()

    def openCSV(self):
        filename = filedialog.askopenfilename(filetypes=[("Текстовый файл", "*.csv")])
        self.treeviewCSV(filename)
        self.info(filename, os.path.getsize(filename), time.ctime(os.path.getmtime(filename)))
        self.logic.get_prediction(filename)

    def info(self, filename, size, time):
        Label(text = 'Версия: 0.1 beta    Элементов: ' + str(self.infoStr) + '     ' +\
                     'Файл: ' + str(filename.split('/')[-1]) + '     ' +\
                     'Размер: ' + str(round(size/1024, 1)) + ' Кб' + '     ' + \
                     'Изменен: ' + str(time),  bd=1, relief=SUNKEN, anchor=W) \
        .grid(row=6, column=0, sticky=N+S+W+E, columnspan=6, pady = 20)

    def treeviewCSV(self, filename):
        self.tree.delete(*self.tree.get_children())
        with open(filename, newline="") as f:
            for i, date in enumerate(csv.reader(f)):
                if i == 0:
                    continue
                data_mod = [i] + date
                #data_mod = [i] + list(map(float, date))
                #round5 = lambda x: round(x, 10)
                #self.tree.insert("", END, values = list(map(round5, data_mod)))
                if i % 2 == 0:
                    self.tree.insert("", END, values = data_mod, tag='gray')
                else:
                    self.tree.insert("", END, values = data_mod)

            self.logic.calc_graph(list(map(float, date)))
            self.infoStr = i

    def print_selection(self, event):
        item = self.tree.selection()[0]
        for selection in self.tree.selection():
            self.item = self.tree.item(selection)
            values = list(map(float, self.item["values"]))
            values.pop(0)
            self.logic.calc_graph(values)

    def graphMat(self, conc, conc1, conc2, profit):
        plt.clf()
        plt.plot(self.gas, conc1, 'r', label='Предельно допустимое значение')
        plt.plot(self.gas, conc2, 'y', label='Допустимое значение')
        plt.plot(self.gas, conc, 'g', label='Текущие значения')
        plt.fill_between(self.gas, conc1, profit, color='r', alpha = 0.05)
        plt.fill_between(self.gas, profit, conc2, color='y', alpha=0.05)
        plt.grid()
        plt.legend()
        plt.xticks(range(4), ['H2', 'CO', 'C2H4', 'C2H2'])
        plt.title("Хроматографический анализ трансформаторного масла")
        plt.ylabel('Объемная концентрация')
        self.fig.canvas.draw()

        #fig_ax_1 = fig.add_subplot(gs[0, :])
        #plt.plot(gas, conc, conc1)
        #plt.xticks(range(4), ['H2', 'CO', 'C2H4', 'C2H2'])
        #plt.title("Хроматографический анализ трансформаторного масла (ХАРГ)")

        #fig_ax_2 = fig.add_subplot(gs[1, 0])
        #plt.plot(gas, conc)
        #plt.title("PD")

        #fig_ax_3 = fig.add_subplot(gs[1, 1])
        #plt.plot(gas, conc)
        #plt.title("LED")

        #fig_ax_4 = fig.add_subplot(gs[1, 2])
        #plt.plot(gas, conc)
        #plt.title("LTO")

        #plt.subplot(1, 3, 1)
        #plt.plot(gas, conc, conc1)
        #plt.xticks(range(4), ['H2', 'CO', 'C2H4', 'C2H2'])
        #plt.title("Частичный разряд (PD)")

        #plt.subplot(1, 3, 2)
        #plt.plot(gas, conc)
        #plt.xticks(range(4), ['H2', 'CO', 'C2H4', 'C2H2'])
        #plt.title("Разряд низкой энергии (LED)")

        #plt.subplot(1, 3, 3)
        #plt.plot(gas, conc)
        #plt.xticks(range(4), ['H2', 'CO', 'C2H4', 'C2H2'])
        #plt.title("Низкотемпературный термический дефект (LTO")

    def set_prediction(self, defect_label, time_forecast):
        self.predict_label.config(text = 'Прогноз по работе трансформатора: ' + defect_label + '.')
        self.time_forecast.config(text = 'Достигнет предельно-допустимой концентрации через: ' + str(time_forecast / 2) + ' суток.')


    def set_logic(self, logic):
        self.logic = logic

















