from urllib.request import urlopen
import urllib.request
import urllib
import urllib3
import os
import re
import tarfile

#os.chdir(os.d)#To change the current path
#mainpath='http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/'
mainpath = 'http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Original/48kHz_16bit/'
def wav_files(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] == ".wav":
            yield tarinfo
def gettgz(url):
    page=urlopen(url)
    html=page.read().decode('utf-8')
    reg=r'href=".*\.tgz"'
    tgzre=re.compile(reg)
    tgzlist=re.findall(tgzre,html)  #Find all of the.Tgz files
    for i in tgzlist:
        filename=i.replace('href="','')
        filename=filename.replace('"','')
        if(not os._exists(filename)):

            print ('Downloading:'+filename) # prompts the file being downloaded
            downfile=i.replace('href="',mainpath)
            downfile=downfile.replace('"','') #Each file integrity
            req = urllib.request.Request (downfile)  #Download the file
            ur =  urlopen(req).read()
            open(filename,'wb').write(ur)
            #Download the files stored in tgz format on the D disc
            tar = tarfile.open(filename)
            tar.extractall(members=wav_files(tar))
            print(  filename+ ' extracted *****')
            tar.close()
            os.remove(filename)
    #    refiles.write(downfile+'\n')
#refiles.close()

if __name__ == '__main__':
    html = gettgz(mainpath)
