import numpy as np

class Evaluation():
    def __init__(self,source_hashs,source_labels,goal_hashs,goal_labels,label_compare=1):
        self.shs = source_hashs
        self.sls = source_labels
        self.ghs = goal_hashs
        self.gls = goal_labels
        self.lc = label_compare

    def __check_common_divisor(self, a, b):
        for i in range(2, min(a,b) + 1):
            if a % i == 0 and b % i == 0:
                return True
        return False

    def __cal_elements_dis(self, a, b, method='Hamming2'):
        if method == 'Hamming2':
            return bin(int(a, 2) ^ int(b, 2)).count('1')

    def __check_label(self, a, b):
        if self.lc == 1:
            return a == b
        if self.lc == 2:
            return self.__check_common_divisor(a, b)

    def cal_dis_list(self, gh):
        hamdiss = []
        for sh in self.shs:
            hamdiss.append([self.__cal_elements_dis(gh, sh[0])])
        return hamdiss

    def sort_for_label(self,hamdis_label,goal_cls):
        shot = []
        for i in range(0,len(hamdis_label)):
            if self.__check_label(hamdis_label[i][1], goal_cls):
                shot.append([0])
            else:
                shot.append([1])
        hamdis_label_shot = np.concatenate((hamdis_label, shot), axis=1)
        hamdis_label_shot = hamdis_label_shot[hamdis_label_shot[:, 2].argsort()]

        return hamdis_label_shot.T[:-1].T

    def cal_AP_score(self,hamdis_label,goal_cls,R):
        shot = 0
        score = 0
        for i in range(0,R):
            if self.__check_label(hamdis_label[i][1], goal_cls):
                shot = shot + 1
                score = score + shot/(i+1)
        return score/R


    def cal_P_score(self,hamdis_label,goal_cls,HR):
        shot = 0
        sample = 0
        while hamdis_label[sample][0] <= HR:
            if self.__check_label(hamdis_label[sample][1], goal_cls):
                shot = shot + 1
            sample = sample + 1
        if sample == 0:
            return 0
        return shot/sample

    def cal_R_score(self,hamdis_label,goal_cls,clsnum,proportion):
        score = np.zeros(len(proportion))
        shot = 0
        sample = 0
        pro_index = 0
        pros = clsnum * np.array(proportion)
        while shot < clsnum:
            if self.__check_label(hamdis_label[sample][1], goal_cls):
                shot = shot + 1
            sample = sample + 1
            if shot == pros[pro_index]:
                score[pro_index] = shot/sample
                pro_index = pro_index + 1
        return score

    def get_mAP(self,R,best=True):
        score = 0
        for i in range(0,len(self.ghs)):
            hamdiss = self.cal_dis_list(self.ghs[i][0])
            hamdis_label = np.concatenate((hamdiss, self.sls), axis=1)
            if best==True:
                hamdis_label = self.sort_for_label(hamdis_label,self.gls[i][0])
            hamdis_label = hamdis_label[hamdis_label[:, 1].argsort()]
            score = score + self.cal_AP_score(hamdis_label,self.gls[i][0],R)
        return score/len(self.ghs)

    def get_Precision_HR(self,HR):
        score = 0
        for i in range(0,len(self.ghs)):
            hamdiss = self.cal_dis_list(self.ghs[i][0])
            hamdis_label = np.concatenate((hamdiss, self.sls), axis=1)
            hamdis_label = hamdis_label[hamdis_label[:, 1].argsort()]
            score = score + self.cal_P_score(hamdis_label, self.gls[i][0], HR)
        return score / len(self.ghs)

    def get_PrecisionRecall(self,
                            eachclsnum=[1000,1000,1000,1000,1000,1000,1000,1000,1000,1000],
                            proportion=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                            best=True):
        score = np.zeros(len(proportion))
        for i in range(0, len(self.ghs)):
            hamdiss = self.cal_dis_list(self.ghs[i][0])
            hamdis_label = np.concatenate((hamdiss, self.sls), axis=1)
            if best == True:
                hamdis_label = self.sort_for_label(hamdis_label, self.gls[i][0])
            hamdis_label = hamdis_label[hamdis_label[:, 1].argsort()]
            score = score + self.cal_R_score(hamdis_label, self.gls[i][0], eachclsnum[self.gls[i][0]],proportion)
        return score / len(self.ghs)
