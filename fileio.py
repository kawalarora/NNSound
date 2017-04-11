import os
import random
class fileio :
    def __init__(self,classtype):
        self.folder=os.getcwd()+"/trainingfiles/" + classtype
        #self.folder="/users/kawal/temp/doggytraining/set1"
        self.TrainingFileFolder=self.folder
    def GetCounter(self):
        f = open(self.folder+ "/train/counter.txt", 'r+')
        self.counter=int(f.read())
        return self.counter
    def UpdateCounter(self):
        f = open(self.folder + "/train/counter.txt", 'w+')
        self.counter=self.counter+ 1
        f.write(str(  self.counter))
        f.close()
    def WriteStatus(self,  filename,status):
        f = open(filename, 'w+')
        f.write(status)
        f.close()

    def WriteNextWav(self,file,tag,status):
        self.GetCounter()
        rn=random.randint(1, 100)
        wavfilename= self.folder+  "/train/Record_{0}_{2}_{1}_{3}.wav".format(  tag , str(self.counter),rn,status)
        file.save(wavfilename)
        self.UpdateCounter()

    def WriteClassifyWav(self,file):
        self.GetCounter()
        wavfilename = self.folder + "/lastRun/Record_{0}.wav".format(  str(self.counter) )
        file.save(wavfilename)
        self.UpdateCounter()

        return wavfilename



if __name__ =="__main__":
    file=fileio()
    print(file.GetCounter())
    file.UpdateCounter()
    print(file.GetCounter())
    file.UpdateCounter()
    print(file.GetCounter())
    file.UpdateCounter()



