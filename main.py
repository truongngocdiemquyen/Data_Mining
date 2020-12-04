from tkinter import *
from tkinter import filedialog
from tools import dataloader, model_team_7 
import numpy as np
from time import time

def timer(func):
    def wrapper(*args, **kwargs):
        tic = time()
        func(*args, **kwargs)
        args[0].label1['text'] += '\ntime: %.2f (s)'% (time() - tic)
    return wrapper


class App(Frame):
    """docstring for App_TomatoTimer"""
    def __init__(self, master):
        self.cfg = {}
        cfg_normal = {
            'isNormal' : True,
            'type': 'min_max' , # z_score, min_max
            'min_max': [-1, 1]
        }

        config = {}
        config['path_stopwords'] = 'data/stopwords_en.txt'

        config['random_state'] = 432751
        config['train_test']= '7/3'
        config['check_split_data'] = False
        
        config['cfgNormal'] = cfg_normal
        config['impute_numeric'] ='mean'
        config['impute_object'] ='most_frequent'

        config['num_cluster'] = 8
        self.cfg = config

        super(App, self).__init__()
        windowSize = (master.winfo_screenwidth(), master.winfo_screenheight())
        size = (windowSize[1]-100, windowSize[1] )

        master.geometry("{}x{}+{}+0".format(size[0], size[1], windowSize[0] - size[1]))
        master.maxsize(size[0],size[1])
        master.minsize(size[0],size[1])
        master['bg'] = 'SkyBlue1'

        Frame_top = Frame(master, bg='SkyBlue1',width= windowSize[0])
        Frame_top.pack(side=TOP, fill = "both")


        Frame_top_1 = Frame(Frame_top, bg='SkyBlue1',)
        Frame_top_1.pack(side=LEFT,  padx=20,  pady=20)
        self.btn_choose_file = Button(Frame_top_1, text='Choose file', command=self.choose_file)
        self.btn_choose_file.pack(side=TOP, fill = "both")

        self.label_choose_file = Label(Frame_top_1, width = 40)
        self.label_choose_file.pack()
        self.btn_load_pre_data = Button(Frame_top_1, text='Load preprocessed data', command=self.load_pre_data)
        self.btn_load_pre_data.pack(side=TOP, fill = "both", pady=20)

        Frame_top_2 = Frame(Frame_top, bg='SkyBlue1',)
        Frame_top_2.pack(side=LEFT,  padx=20,  pady=20)
        self.temp_1(Frame_top_2)
        self.btn_choose_model = Button(Frame_top_2, text='Choose file save algorithm', command=self.choose_model)
        self.btn_choose_model.pack(side=TOP, fill = "both", pady =10)

        Frame_tleft_1 = Frame(Frame_top_2, bg='SkyBlue1', )
        Frame_tleft_1.pack(side=LEFT)

        Frame_top_3 = Frame(Frame_top, bg='SkyBlue1')
        Frame_top_3.pack(side=LEFT, fill = "both", padx=10, pady=20)
        

        # Frame_1_1 = Frame(Frame_tright)
        # Frame_1_1.pack()
        self.btn_pre_data = Button(Frame_top_3, text='Preprocess data', command=self.preprocess_data)
        self.btn_pre_data.pack(side=TOP, fill = "both")

        self.btn_action = Button(Frame_top_3, text='Train algorithm', command=self.train)
        self.btn_action.pack(fill = "both", pady=3)

        self.btn_evaluation_model = Button(Frame_top_3, text='Evaluation algorithm', command=self.evaluation_model)
        self.btn_evaluation_model.pack(fill = "both", pady=3)

        self.btn_save_model = Button(Frame_top_3, text='Save algorithm', command=self.save_model)
        self.btn_save_model.pack(fill = "both", pady=3)

        Frame_predict = Frame(Frame_top_3, bg='Blue')
        Frame_predict.pack(side=TOP, fill = "both",  pady=5)
        self.btn_Choose_file_to_predict = Button(Frame_predict, bg='Green', fg='White',text='Choose file and predict', command=self.Choose_file_and_predict)
        self.btn_Choose_file_to_predict.pack( fill = "both", pady=3)

        # botttt
        self.Frame_bot = Frame(master, bg='SkyBlue1')
        self.Frame_bot.pack(side=BOTTOM)

        # n, ne, e, se, s, sw, w, nw, or center

        self.label1 = Label(self.Frame_bot, text='',width=windowSize[0], height=int(windowSize[1]), anchor="nw",font=("Arial", 15))
        self.label1.pack()





        # btn_choose_file = Button(master)
    

    def Choose_file_and_predict(self):
        testpath = filedialog.askopenfilename()
        if testpath == '': 
            return
        self.label_choose_file['text'] = testpath.split('/')[-1]
        X = self.data.predict(testpath)
        self.label1['text'] = 'Predict:\n'
        label = self.model.predict(X)
        print(testpath)
        path = testpath.split('/')
        path[-1] = 'label_' + path[-1] + '.txt'
        label = [str(i) for i in label]
        print(label)
        string = '\n'.join(label)
        path = '/'.join(path)
        with open(path, 'w+') as f:
            f.write(string)
   
    @timer
    def train(self):
        self.model = model_team_7(self.cfg)
        self.model.fit(self.data)
        self.label1['text'] = 'Train {}'.format(self.cfg['algorithm'])
    
    @timer  
    def evaluation_model(self):
        train_set, test_set =  (self.data.X_train, self.data.y_train ), (self.data.X_test,  self.data.y_test)

        self.label1['text'] = self.model.evaluation(train_set, test_set)
    
    def choose_model(self):
        self.model = model_team_7(self.cfg)
        path = filedialog.askopenfilename()
        if path == '': 
            return
        print(path.split('/')[-1])
        self.label1['text'] = 'load %s' % path.split('/')[-1]
        self.model.load(path)

    def save_model(self):
        self.label1['text'] = self.model.save()

    def temp_1(self, frame):
        self.v = IntVar()
        self.v.set(0) 
        model = [
            ("SVM",1),
            ("Decision Tree",2),
            ("Logistic Regression",3),
            ("Random Forest",4),
            ("Naive Bayes",5)
        ]

        Label(frame, text="""Choose algorithm:""", bg='Orange').pack(fill='both', anchor=W)
        self.cfg['algorithm'] ='svm'

        for val, language in enumerate(model):
            Radiobutton(frame, 
                  text=language[0],
                  indicatoron = 0,
                  width = 20,
                  padx = 20, 
                  variable=self.v, 
                  command=self.choose_al,
                  value=val).pack(anchor=W, pady=2)

    def choose_al(self):
        names = ['svm', 'decision_tree','LogisticRegression', 'random', 'naive']
        self.cfg['algorithm'] = names[self.v.get()]
        print(names[self.v.get()])

    def format_print(self, data):
        text = ''

        for col in data.columns:
            text += '{} type: {}, '.format(col, data[col].dtype)
            arr = data[col].describe().unique()
            if data[col].dtype == 'object':

                text += 'most frequent: {} ({}), unique:{}\n'.format(arr[2], arr[3], arr[1])
            else: 
                arr = [round(i, 2) for i in arr]
                text += 'min: {}, max: {}, mean: {} std: {}\n'.format(arr[3], arr[-1], arr[1], arr[2])
        return text
   
    @timer 
    def choose_file(self):
        self.file_name = filedialog.askopenfilename()
        self.label_choose_file['text'] = self.file_name.split('/')[-1]
        if self.file_name == '': 
            return
        self.cfg['path_data'] = self.file_name
        self.data = dataloader(self.cfg)
        self.data.read_data()
        self.label1['text'] = str(self.data.origin_df.shape) + '\n' + self.format_print(self.data.origin_df)

    @timer
    def preprocess_data(self):

        self.label1['text'] = 'Processing \nMissing Value\nChange nominal to numeric\n'


        self.data.transform()

        # normalize 
        if  self.data.cfg['cfgNormal']['isNormal']:
            self.data.data = self.data.normalize(self.data.data)

        self.data.split_data()
        a, b = len(self.data.X_train) , len(self.data.X_test)
        self.label1['text'] +=  'Train/test: {}, {} ({}/{})\n'.format(a, b, int(a/(a+b)*100+0.5), int(b/(a+b)*100+0.5))
        
        self.label1['text'] += self.data.remove_outlier() + '\n'


        self.label1['text'] += str(self.data.data.head(5))#self.format_print(self.data.data)  
        self.data.save()   
        self.label1['text'] += '\ndata preprocessing saved' 
    
    def load_pre_data(self):
        self.data = dataloader(self.cfg)
        self.data.load()
        self.label1['text'] = 'Train: {}, Test: {}\n'.format(self.data.X_train.shape, self.data.X_test.shape)
        for i in range(self.data.X_train.shape[1]):
            temp = self.data.X_train[:, i]
            self.label1['text'] += 'col: %d, Min: %.3f, Max: %.3f, Mean: %.3f, Std:%.3f\n' %(i, temp.min(), temp.max(), temp.mean(), temp.std())

def main():
    root = Tk()
    app = App(master=root)
    
    app.mainloop()
if __name__ == '__main__':
        main()    

