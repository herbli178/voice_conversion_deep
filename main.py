from feature import featureExtractor
import numpy as np
from tqdm import tqdm
import os 
from config import config_all
from statistics import generate_statistics_feature,GV_estimate
from jointfeature import gen_join_features,static_delta
import joblib
from model import GMMTrainer,GMMConvertor,f0_convert,GV_postfilter,synthesis
from scipy.io import wavfile

import pdb
import time
import matplotlib.pyplot as plt


def feature_collect(spk, files, config,feature_path):
    for file in tqdm(files, total = len(files)):
        # 计算WAV的特征
        f0, mcep, npow ,ap = featureExtractor(file,config)
        # 以字典的形式保存
        feature = {}
        feature["f0"] = f0
        feature["mcep"] = mcep
        feature["npow"] = npow
        file_name = spk + "_" +file.split("/")[-1].split(".")[0] + ".npy"
        np.save(os.path.join(feature_path,file_name),feature)
        
# 测试程序    
if __name__ == "__main__":        
    
    # pdb.set_trace()
    # 读取 训练语音对
    pair_scp = "train_pair_0.scp"
    # pair_scp = "train_pair_2.scp"
    with open(pair_scp,'r') as f:
        lines = f.read().splitlines()
    lines = [line.split() for line in lines]
        
    spk1 = np.unique(np.array([line[0] for line in lines]))[0]
    spk2 = np.unique(np.array([line[2] for line in lines]))[0]
    print(spk1)
    print(spk2)
    files_spk1 = [line[1] for line in lines]
    files_spk2 = [line[3] for line in lines]
    
    
    
    # print("step 1 feature collect")
    # feature_path = config_all["path_fea"]
    # os.makedirs(feature_path,exist_ok=True)
    # # 对spk1的特征进行采集 
    # feature_collect(spk1, files_spk1,config_all["Feature"],feature_path)
    # # 对spk2的特征进行采集
    # feature_collect(spk2, files_spk2,config_all["Feature"],feature_path)
  
    
    # print("step 2 compute static information")
    # static_path = config_all["path_model"]
    # os.makedirs(static_path,exist_ok=True)
    # spks = [spk1, spk2]
    # files_spks = [files_spk1,files_spk2]
    # feature_path = config_all["path_fea"]
    # for spk,files in zip(spks,files_spks):
    #     list_fea = [os.path.join(feature_path,spk + "_" +file.split("/")[-1].split(".")[0] + ".npy") for file in files]
    #     f0stats,gvstats = generate_statistics_feature(list_fea)
    #     fea_static = {}
    #     fea_static["f0stats"] = f0stats
    #     fea_static["gvstats"] = gvstats
    #     save_name = os.path.join(static_path,spk+'.npy')
    #     np.save(save_name,fea_static)
     
    
    # # pdb.set_trace()
    # print("step 3 get joint mcep feature")
    # fea_file_list_spk1 = [spk1 + "_" +file.split("/")[-1].split(".")[0] + ".npy" for file in files_spk1]
    # fea_file_list_spk2 = [spk2 + "_" +file.split("/")[-1].split(".")[0] + ".npy" for file in files_spk2]
        
    # feature_path = config_all["path_fea"]
    # jnt_mcep_data = gen_join_features(fea_file_list_spk1,
    #                                   fea_file_list_spk2,
    #                                   feature_path, 
    #                                   config_all["GMM-mcep"])
    
    # np.save(os.path.join(feature_path,"jnt_mecp_data.npy"),jnt_mcep_data)
    
    
    # print("step 4 train GMM model")
    
    # feature_path = config_all["path_fea"]
    # jnt_mcep_data = np.load(os.path.join(feature_path,"jnt_mecp_data.npy"))
    
    # model_path = config_all["path_model"]
    # os.makedirs(model_path,exist_ok=True)
    
    
    # mecp_GMM_config = config_all["GMM-mcep"]
    # gmm_mecp = GMMTrainer(n_mix=mecp_GMM_config["n_mix"] ,
    #                       n_iter=mecp_GMM_config["n_iter"])
    # gmm_mecp.train(jnt_mcep_data)
    # joblib.dump(gmm_mecp.param, os.path.join(model_path,"gmm_mecp.pkl"))
    # print("Save gmm_mecp")
    

    
    # print("step 5 computr static information of trained convered features")
    
    
    # mecp_GMM_config = config_all["GMM-mcep"]
    # mecp_convertor = GMMConvertor(n_mix=mecp_GMM_config["n_mix"])
    
    
    # model_path = config_all["path_model"]
    # param = joblib.load(os.path.join(model_path,"gmm_mecp.pkl"))
    # mecp_convertor.open_from_param(param)
    
    
    # fea_file_list_spk1 = [spk1 + "_" +file.split("/")[-1].split(".")[0] + ".npy" for file in files_spk1]
    # sd = mecp_GMM_config["sd"]
    # cv_mceps =[]
    # feature_path = config_all["path_fea"]
    # for file in fea_file_list_spk1:
    #     feature = np.load(os.path.join(feature_path,file),allow_pickle=True).item()
    #     mcep = feature["mcep"]
    #     mcep_0th = mcep[:, 0]
    #     cvmcep = mecp_convertor.convert(static_delta(mcep[:, sd:]),
    #                            cvtype=mecp_GMM_config["cvtype"])
    #     cvmcep = np.c_[mcep_0th, cvmcep]
    #     cv_mceps.append(cvmcep)
    
    
    # cvgvstats = GV_estimate(cv_mceps)
    # model_path = config_all["path_model"]
    # save_name = os.path.join(model_path,spk1+"2"+spk2+"_cvgvstats"+".npy")
    # np.save(save_name,cvgvstats)
    
    
    print("step 6 conver a wav file")
   
    mecp_GMM_config = config_all["GMM-mcep"]
    mecp_convertor = GMMConvertor(n_mix=mecp_GMM_config["n_mix"])
    
    
    model_path = config_all["path_model"]
    param = joblib.load(os.path.join(model_path,"gmm_mecp.pkl"))
    mecp_convertor.open_from_param(param)
    
    static_path = config_all["path_model"]
    
    org_static = np.load(os.path.join(static_path,spk1+".npy"),allow_pickle=True).item()
    orgf0stats = org_static["f0stats"]
   
    
    tar_static = np.load(os.path.join(static_path,spk2+".npy"),allow_pickle=True).item()
    tarf0stats = tar_static["f0stats"]
    targvsats = tar_static["gvstats"]
    
    
    cvgvstats = np.load(os.path.join(static_path,spk1+"2"+spk2+"_cvgvstats"+".npy"))

    counter = 0
    times = []
    file_durations = []
    
    with open('eval_vcc2sf1.txt') as f:
        for line in f:
            org_wav = line[:-1]
            # org_wav = "/Users/herbertli/Downloads/DS_10283_3061/vcc2018_evaluation/VCC2SF1/30003.wav"
            source_rate, source_sig = wavfile.read(org_wav)
            duration_second = len(source_sig) / float(source_rate)
            file_durations.append(duration_second)
            t1 = time.time()
            
            f0, mcep, npow ,ap = featureExtractor(org_wav,config_all["Feature"])
            mcep_0th = mcep[:, 0]
            
            
            cvf0 = f0_convert(f0, orgf0stats, tarf0stats)
            
            
            cvmcep_wopow = mecp_convertor.convert(static_delta(mcep[:, 1:]),
                                                cvtype=mecp_GMM_config["cvtype"])
            cvmcep = np.c_[mcep_0th, cvmcep_wopow]
            
            
            cvmcep_wGV = GV_postfilter(cvmcep,
                                    targvsats,
                                    cvgvstats=cvgvstats,
                                    alpha=config_all["GV_morph_coeff"],
                                    startdim=1)
                                        
            
            wav = synthesis(cvf0,cvmcep_wGV,ap,config_all["Feature"])
            
            wav = np.clip(wav, -32768, 32767)
            t2 = time.time()
            times.append(t2-t1)
            # print(times)
            filepath = os.path.join("output/", ("conv_2sm1_400" + str(counter) + ".wav"))
            counter += 1
            wavfile.write(filepath, config_all["Feature"]["fs"], wav.astype(np.int16))
  
    
    print("time taken to run voice conversions --------", times)
    print("length of wav files ---------", file_durations)

    plt.plot(file_durations, times)
    plt.xlabel('wav file durations (s)')
    plt.ylabel('voice conversion times (s)')
    plt.savefig('voice_conversion_times.png')
    plt.close()

    
    
    
    
    
    
    
    
    