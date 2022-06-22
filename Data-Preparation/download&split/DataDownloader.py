'''
  @InProceedings{zhao2018sound,
        author = {Zhao, Hang and Gan, Chuang and Rouditchenko, Andrew and Vondrick, Carl and McDermott, Josh and Torralba, Antonio},
        title = {The Sound of Pixels},
        booktitle = {The European Conference on Computer Vision (ECCV)},
        month = {September},
        year = {2018}
  }
'''

#!pip install pytube

#import pytube
import json
from pytube import YouTube
import os

class DataDownloader():
  def __init__(self):
    pass


  def notContaining(self, path, title):

    for file in os.listdir(path):
      if file == title:
        return False
    return True

  def downloadSingleVideo(self, path, videoCode, i, type):
    link = "https://www.youtube.com/watch?v=" + videoCode
    try:
      yt = YouTube(link)
      name = yt.title
    except:
      print("Connection Error with url '" + link + "'")
      name = "$/$"

    #print("Title: ",yt.title)
    vid = None
    errorLog = ""

    #vid = yt.streams.get_by_itag(18).download(path)

    if self.notContaining(path, name):
      
      try:
        vid = yt.streams.get_by_itag(18).download(path)
        print("Done - " + str(i))
      except:
        errorLog = "  Error occured downloading video number " + str(i) + " of " + type + "-Set with url '" + link + "'\n"
        print("error - " + str(i))

    return vid, errorLog

  def downloadDataFromJSON(self, path, jsonObj, n, type):
    i = 1
    errorLog = ""
    for node in jsonObj:
      if node == "videos":
        for key in jsonObj[node]:
          urls = jsonObj[node][key]
          label = key
          print(label + " Videos:")
          try:
            os.mkdir(path + "/" + key)
          except:
            pass

          for url in urls:
            print("downloading video ", i, " out of ",n)
            _, eL = self.downloadSingleVideo(path + "/" + key, url, i, type)
            errorLog += eL
            i = i + 1

    return errorLog

#if __name__ == 'main':
if 1 == 1:

  Directory_Path = "/dsi/gannot-lab/datasets/Music_arme"

  try:
    os.mkdir(Directory_Path + "/Data")
  except:
    pass
  
  try:
    os.mkdir(Directory_Path + "/Data/Solo")
  except:
    pass
  

  DD = DataDownloader()


  #Solo Music - 536 urls, 11 classes
  path = Directory_Path + "/Data/Solo"
  jsonObj = json.load(open(Directory_Path + '/MUSIC/MUSIC_Solo.json'))
  
  errorLogSolo = DD.downloadDataFromJSON(path, jsonObj, 536, "Solo")
  
  # #Solo21 Music - 1164  urls, 21 classes
  # path = "Data/Solo21"
  # jsonObj = json.load(open(Directory_Path + '/MUSIC/MUSIC_Solo21.json'))
  
  # errorLogSolo21 = DD.downloadDataFromJSON(path, jsonObj, 1164, "Solo21")


  try:
    os.mkdir(Directory_Path + "/Data/Duet")
  except:
    pass

  #Duet Music - 149 urls, 9 classes
  path = Directory_Path + "/Data/Duet"
  jsonObj = json.load(open(Directory_Path + '/MUSIC/MUSIC_Duet.json'))

  errorLogDuet = DD.downloadDataFromJSON(path, jsonObj, 149, "Duet")

  try:
    os.mkdir(Directory_Path + "/Logs")
  except:
    pass

  try:
    file = open(Directory_Path + "/Logs/SoloErrorsLog.txt", "x")
  except:
    file = open(Directory_Path + "/Logs/SoloErrorsLog.txt", "a")

  file.write("\nSolo errors : \n")
  file.write(errorLogSolo)

  # try:
  #   file = open("Logs/SoloErrorsLog21.txt", "x")
  # except:
  #   file = open("Logs/SoloErrorsLog21.txt", "a")

  # file.write("\nSolo21 errors : ")
  # file.write(errorLogSolo21)


  try:
    file = open(Directory_Path + "/Logs/DuetErrorsLog.txt", "x")
  except:
    file = open(Directory_Path + "/Logs/DuetErrorsLog.txt", "a")

  file.write("\nDuet errors : \n")
  file.write(errorLogDuet)


  print('Done!')