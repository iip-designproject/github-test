"""
Drum Data
 1. 10초짜리 빈 음원 생성
 2. main_shots 만들기
 3. mixed_shots 만들기

"""



"""
[data mixing코드 7/18 피드백]

1. 오다시티로 봤을 때 아무것도 없어보이는 부분 자르기
->용량 줄이고 겹치는것 방지 -> '완료'

2. main shot에 여러 악기 선정하기
-> 10초내에 tom, snare, hihat이런식으로 -> '완료'

3. 실제 드럼배치 고려하기 -> '완료'

4. shot끼리 겹치지 않게 배치 -> '완료'

5. .txt 파일 만들기 -> '완료'

"""





import numpy as np
import wave, math, struct
import random

import torch
import torchaudio
from pathlib import Path
from torchaudio.utils import download_asset
import os


'''직접 입력하는 부분'''
dataset_path = '/home/dp2023/JaeeunB/dataset'
oneshot_path = '/home/data/dp2023/Prelude_Studio/ONE_SHOT'
oneshot_t_path = dataset_path + '/oneshot_t'
num_data = 100 #만들고 싶은 data 개수


""" 1. 10sec 짜리 빈 음원 생성"""

sample_Rate = 48000
n_samples = 5 * sample_Rate
n_channels = 1

output_file = dataset_path + '/empty_10sec.wav'

with wave.open(output_file, "wb") as file:
    file.setnchannels(n_channels)
    file.setsampwidth(1)
    file.setframerate(sample_Rate)
    for i in range(n_samples):
        value = 0
        data = struct.pack('<h', value)
        file.writeframesraw(data)
    file.close()

waveform, samplerate = torchaudio.load(output_file)










""" 2. dataset 저장할 폴더 생성"""


#main파일 불러오기
path = dataset_path + "/oneshot_t" #다듬어진  
if(os.path.isdir(path)==False):
    os.mkdir(path)


path = dataset_path + "/main_shots"
if(os.path.isdir(path)==False):
    os.mkdir(path)


path = dataset_path + "/noise_shots"
if(os.path.isdir(path)==False):
    os.mkdir(path)
 








""" 3. wav파일에서 소리 안나는 부분 자르기"""       



### 1) /oneshot 폴더 전체 wav파일 list만들기 ###
folder_list = os.listdir(oneshot_path) #하위 폴더이름 나열
wav_name = []
wav_paths = []

for folder_idx in range(len(folder_list)):
    folderpath = oneshot_path + '/' + folder_list[folder_idx] +'/'
    wav_name = os.listdir(folderpath)
    wav_name.extend(wav_name)




### 2) 수정 wav파일 저장할 폴더 생성###
    path = dataset_path + "/oneshot_t" + '/' + folder_list[folder_idx]
    if(os.path.isdir(path)==False):
        os.mkdir(path)
###


    for wav_idx in range(len(wav_name)):
        wav_path = folderpath + wav_name[wav_idx] ##wav_name : 해당 폴더의 wav파일 이름만 list
        wav_paths.append(wav_path) ##wav_paths : 모든 wav파일 경로 list


### 2) wav파일에서 소리 안나는 부분 자르기 ###
        original_waveform, sample_rate = torchaudio.load(wav_paths[wav_idx], format = 'wav')

        for i in range(original_waveform.size(1)):
            if (original_waveform[0][i] > 0.00000001):
                original_waveform = original_waveform[:, i:]
                break
        trimmed_path = dataset_path + '/oneshot_t/' + folder_list[folder_idx] + '/' + wav_name[wav_idx] + '_t.wav'
        torchaudio.save(trimmed_path, original_waveform, sample_rate)

print("one shot 파일 총 개수: ", len(wav_paths))








""" 4. main.wav 만들기"""

### main_shots.txt 파일 만들고, 이미 있으면 초기화###
f = open( dataset_path + '/main_shots.txt', 'w')
f.close()
with open(dataset_path + '/main_shots.txt', 'r+') as f:
    f.truncate(0)


##### main shots으로 만들 wav파일 하나씩 가져오기 #####

for i in range(num_data): #len(path_list)
    empty_wf, sr = torchaudio.load(dataset_path + '/empty_10sec.wav')
    main_shots = empty_wf

    folder_idx = random.randint(0, len(folder_list) -1 )  ### 악기 고르기 (폴더 고르기)
    folderpath = oneshot_t_path + '/' + folder_list[folder_idx] + '/'
    wav_name = os.listdir(folderpath)
    wav_name.extend(wav_name) ### wav_name : 해당 폴더의 wav파일 이름만 list

    n_shot = 0
    amp_time = torch.FloatTensor([[1.0, 1.0, 1.0 ], [1.0, 1.0, 1.0]])

    wav_paths = []

    for wav_idx in range(len(wav_name)):
        wav_path = folderpath + wav_name[wav_idx]
        wav_paths.append(wav_path) ### wav_paths : 해당 폴더의 모든 wav파일 경로 list


    while(amp_time.size(1) < (10 * sample_Rate)): #10초짜리 wav파일 만들기
        wav_idx = random.randint(0, len(wav_paths) -1 ) ### wav파일 고르기
        dir_path = wav_paths[wav_idx]
        main_waveform, sample_rate = torchaudio.load(dir_path, format = 'wav')
        num_frames = torchaudio.info(dir_path).num_frames

        num_channels = torchaudio.info(dir_path).num_channels

        if(num_channels !=1 ): #채널 개수 1로 맞추기
            main_waveform = main_waveform[:1, :]


        #amp 조정하기
        A = random.uniform(0.6, 0.9) #main의 amp비율 랜덤으로 정하기
        main_amp = A * main_waveform #main의 amp비율 반영

        timing = []
        #timing 조정하기
        if (n_shot == 0):
            t = random.randint(0, sample_Rate*1) #3초 안에 시작
            timing.append(t)
        else:
            t = random.randint( t , t + sample_Rate*3) #또 3초안에 시작
            timing.append(t)
        
        if ( timing[len(timing)-1] < (timing[len(timing)-2] + num_frames) ): #시간 겹치면
            main_shots = main_shots[:, 0 : timing[len(timing)-2]] #이전에 만든 main_shot 자르기
            pad_zeros = torch.zeros((1, sample_Rate * 10 - main_shots.size(1))) #10초짜리로 맞추기
            main_shots = torch.cat([main_shots, pad_zeros], dim=1)
            

        time_shift =  torch.zeros((1,t)) #랜덤으로 선정한 time shift 반영
        amp_time = torch.cat([time_shift, main_amp], dim=1) #tensor이어붙이기

        #waveform끼리 더하기 위해서 길이 10초로 맞추기
        if (amp_time.size(1) > 10*sample_Rate): #길면 자르고
            amp_time_main = amp_time[:, 0 : sample_Rate * 10]
        else: #짧으면 늘리고
            pad_zeros = torch.zeros((1, sample_Rate * 10 - amp_time.size(1)))
            amp_time_main = torch.cat([amp_time, pad_zeros], dim=1)
        main_shots += amp_time_main
        n_shot += 1



    if main_shots.size(1) > 10*sample_Rate: ###### 10초 넘으면 자르기
        main_shots = main_shots[:, 0 : sample_Rate * 10]


    main_path = dataset_path + '/main_shots' + '/main_' + str(i) +'.wav'
    torchaudio.save(main_path, main_shots, sr)

    # main.wav 무슨 악기로 만들었는지 txt파일로 저장하기
    f = open( dataset_path + '/main_shots.txt', 'a')
    data = folder_list[folder_idx] + '\n'
    f.write(data)
    f.close()








""" 5. noise.wav 만들기"""

main_list = []
main_path = []
path = dataset_path + '/main_shots'
main_list = os.listdir(path) #하위 파일

##### /main_shots 폴더에 있는 전체 wav파일의 경로를 나열한 list만들기 ######

### noise_shots.txt 파일 초기화###
f = open( dataset_path + '/noise_shots.txt', 'w')
f.close()
with open( dataset_path + '/noise_shots.txt', 'r+') as f:
    f.truncate(0)

for i in range(len(main_list)):
    mainpath = path + '/' + main_list[i]
    main_path.append(mainpath)


for i in range(num_data): #여기 괄호에 만들고 싶은 mixed wav개수 넣으면 됨.
    base_wf, sr = torchaudio.load(dataset_path + '/empty_10sec.wav')
    noise_wf = base_wf

    num_noise = random.randint(1, 6) #noise개수

    noise = []
    drum = []
    for j in range(num_noise):
        noise_idx = random.randint(0, len(main_path)-1)
        noise.append(noise_idx)
        f = open( dataset_path + '/main_shots.txt', 'r')
        lines = f.readlines()  
        drum_name = lines[noise_idx] # drum : noise로 쓸 악기 이름 리스트
        drum.append(drum_name)
        f.close()

    f = open( dataset_path + '/noise_shots.txt', 'a')
    for idx in range(num_noise):
        data = drum[idx]
        f.write(data)
    f.write('\n')
    f.close()
    noise_amp = random.uniform(0.5, 0.6)



    #드럼 위치에 따라 amp 조정하기 (마이크는 drum[0]에 위치)
    ###폴더 합치면 이 block은 지우기 ###
    for k in range(0, num_noise):   
        if (drum[k] == 'CRUSH CYMBAL\n'):
            drum[k] = 1
        elif (drum[k] == 'TOM2(middle)\n' or drum[k] == 'BRUSH TOM2(middle)\n'):
            drum[k] = 2
        elif (drum[k] == 'TOM1(high)\n' or drum[k] == 'BRUSH TOM1(high)\n'):
            drum[k] = 3
        elif (drum[k] == 'RIDE CYMBAL\n'):
            drum[k] = 4
        elif (drum[k] == 'HIHAT\n' or drum[k] == 'BRUSH HIHAT\n'):
            drum[k] = 5
        elif (drum[k] == 'BRUSH SNARE1\n' or drum[k] == 'BRUSH SNARE2\n' or drum[k] == 'BRUSH SNARE3\n' or drum[k] == 'SNARE1\n' or drum[k] == 'SNARE2\n' or drum[k] == 'SNARE3\n'):
            drum[k] = 6
        elif (drum[k] == 'TOM3(low)\n' or drum[k] == 'BRUSH TOM3(low)\n'):
            drum[k] = 7
        else:
            drum[k] = 8
    

    #### 요소 개수가 num_noise개가 되도록 noise_amp_list 선언 ##
    noise_amp_list = [0] * num_noise

    for j in range(num_noise):

        if j == 0 :
            noise_amp_list[0] = noise_amp
        else:
            if (drum[0] == 1):
                if (drum[j] == 2 or drum[j] == 5 or drum[j] == 6):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 3 or drum[j] == 8):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 4 or drum[j] == 7):
                    noise_amp_list[j] = noise_amp * 0.3
                elif (drum[j] == 1):
                    noise_amp_list[j] = noise_amp

            if (drum[0] == 2):
                if (drum[j] == 3 or drum[j] == 6):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 1 or drum[j] == 5 or drum[j] == 8 or drum[j] == 7 or drum[j] == 4):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 2):
                    noise_amp_list[j] = noise_amp

            if (drum[0] == 3):
                if (drum[j] == 2 or drum[j] == 8 or drum[j] == 7 or drum[j] == 4):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 1 or drum[j] == 5 or drum[j] == 6):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 3):
                    noise_amp_list[j] = noise_amp

            if (drum[0] == 4):
                if (drum[j] == 3 or drum[j] == 7):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 1 or drum[j] == 2 or drum[j] == 6):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 5 or drum[j] == 8):
                    noise_amp_list[j] = noise_amp * 0.3
                elif (drum[j] == 4):
                    noise_amp_list[j] = noise_amp        
                    
            if (drum[0] == 5):
                if (drum[j] == 1 or drum[j] == 2 or drum[j] == 6):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 3 or drum[j] == 8):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 4 or drum[j] == 7):
                    noise_amp_list[j] = noise_amp * 0.3  
                elif (drum[j] == 5):
                    noise_amp_list[j] = noise_amp

            if (drum[0] == 6):
                if (drum[j] == 3 or drum[j] == 7 or drum[j] == 4):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 1 or drum[j] == 2 or  drum[j] == 5 or drum[j] == 8 ):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 6):
                    noise_amp_list[j] = noise_amp

            if (drum[0] == 7):
                if (drum[j] == 3 or drum[j] == 4 or drum[j] == 8):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 2 or drum[j] == 6):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 1 or drum[j] == 5):
                    noise_amp_list[j] = noise_amp * 0.3  
                elif (drum[j] == 7):
                    noise_amp_list[j] = noise_amp

            if (drum[0] == 8):
                if (drum[j] == 2 or drum[j] == 6 or drum[j] == 7 or drum[j] == 3):
                    noise_amp_list[j] = noise_amp * 0.6
                elif (drum[j] == 1 or drum[j] == 4 or drum[j] == 5):
                    noise_amp_list[j] = noise_amp * 0.4
                elif (drum[j] == 8):
                    noise_amp_list[j] = noise_amp

   # normalization
    if (np.sum(noise_amp_list) > 1):
        noise_amp_list = noise_amp_list / np.sum(noise_amp_list)

    #진짜 만들기
    for noise_idx in range(num_noise):
        wf, noise_sr= torchaudio.load(main_path[noise[noise_idx]], format = "wav") #noise 불러오기

        noised_wf = noise_amp_list[noise_idx] * wf
        noise_wf += noised_wf


    mixed_path = dataset_path + '/noise_shots' + '/noise_' + str(i) + '.wav'
    torchaudio.save(mixed_path, noise_wf, sr)


