import math
import pandas as pd
import numpy as np

class Node:
    def __init__(self, attrs, parent_attr, val):
        self.attrs = attrs
        self.parent_attr = parent_attr
        self.val = val
        self.attr = None
        self.children = []
        self.label = None

class ID3:

    def __init__(self):
        self.root = None
        self.df = None

    @staticmethod
    def H(p, n):
        fp = p / (p + n)
        fn = n / (p + n)
        if fp != 0:
            tp = -fp * math.log(fp, 2)
        else:
            tp = 0
        if fn != 0:
            tn = -fn * math.log(fn, 2)
        else:
            tn = 0
        return tp + tn

    @staticmethod
    def IG(p, n, *args):
        ent = ID3.H(p, n)
        ents = np.array([ID3.H(p_, n_) for p_, n_ in args])
        fracs = np.array([(p_ + n_) / (p + n) for p_, n_ in args])
        dot = ents.dot(-fracs)
        return ent - dot


    def extract_attr(self, n, num_p, num_n):
        print('Extracting best attribute')
        attrs = n.attrs
        best_attr = None
        max_info = 0.0
        best_vals = []
        best_labels = []
        if not attrs:
            return attrs, best_attr, best_vals, best_labels
        for attr in attrs:
            vals = []
            labels = []
            df = self.df[[attr, 'label']]
            df = df.sort_values(by=[attr])
            if False:
                #a = np.array(df)
                #for i in range(1, len(a[0])):
                #    if a[0][i] != a[0][i - 1] and a[1][i] != a[1][i - 1]:
                #        vals.append(a[0][i])
                        # test without this rn            
                print('BADBAD')
            else:
                print('GOODGOOD')
                vals = list(df[attr].unique())
                print(vals)
                a = np.array(df)
                for val in vals:
                    n_pos = len(a[1][a[0] == val & a[1] == 1])
                    n_neg = len(a[1][a[0] == val & a[1] == 0])
                    labels.append((n_pos, n_neg))
            info = self.IG(num_p, num_n, *labels)
            print('testing attr', attr)
            print('ig:', info)
            if info > max_info:
                max_info = info
                best_attr = attr
                best_vals = vals
                best_labels = labels
                print(best_attr, 'set to new best attr')
        attrs.remove(best_attr)
        print(attrs, best_attr, best_vals, best_labels)
        return attrs, best_attr, best_vals, best_labels

    def get_pn(self, parent_attr, val):
        if parent_attr == None:
            p = df['label'].value_counts()[1]
            n = df['label'].value_counts()[0]
            return p, n
        print(df['label'][df[parent_attr] == val])
        p = df['label'][df[parent_attr] == val].value_counts()[1]
        n = df['label'][df[parent_attr] == val].value_counts()[0]
        return p, n

    def add_node(self, attrs, parent_attr, val):
        print('Adding node:', attrs, parent_attr, val)
        n = Node(attrs, parent_attr, val)
        num_p, num_n = self.get_pn(parent_attr, val)
        attrs, attr, vals, labels = self.extract_attr(n, num_p, num_n)
        if not attr:
            n.label = 1 if num_p > num_n else 0
            return n
        n.attr = attr
        for val in vals:
            n.children.append(self.add_node(attrs, attr, val))
        return n

    def learn(self, df):
        self.df = df
        attrs = set(df.columns)
        attrs.remove('label')
        print('Starting')
        self.root = self.add_node(attrs, None, None)


def dataset():
    df = pd.DataFrame({
        'gender': ['Male', 'Female', 'Female', 'Male'],
        'location': ['LA', 'LA', 'LA', 'SD'],
        'label': [1, 0, 0, 1],
    })
    return df

dt = ID3()
df = dataset()
dt.learn(df)













