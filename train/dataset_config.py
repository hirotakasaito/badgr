import os
import argparse
import json
import pprint
from glob import iglob

def main():
    #print("\n" + "==== Add dataset to con")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type=str,default="/root/dataset/sq2/d_kan1/badgr4hz/")
    parser.add_argument("--output-file",type=str,default="/root/BADGR/config/badgr_dirs.json")
    # parser.add_argument("--dataset",type=str,default="/share/private/27th/hirotaka_saito/dataset/recon/")
    # parser.add_argument("--output-file",type=str,default="./config/rssm.json")

    parser.add_argument("--datasets-name",type=str,default = "d_kan1")
    parser.add_argument("--data-len",type=int,default=30)
    args = parser.parse_args()

    data = dict()
    data_list = []
    overwrite = True
    config_file_name = os.path.basename(args.output_file)
    with open(args.output_file) as f:
        config_file = json.load(f)

        for key in config_file:

            if key == args.datasets_name:
                print("\n"+"{} is already in the {}".format(args.datasets_name,config_file_name) + "\n")
                print("Do you want to overwrite ?"+"\n")
                result = input("yes or no\n")
                if result == "no":
                    overwrite = False
                else:
                    data.update(config_file)

    count = args.data_len
    i = 0

    for dataset_path in iglob(os.path.join(args.dataset,"*")):
        # data_list.append(dataset_path)
        if i >= count:
            break
        data_file_path = os.path.basename(dataset_path)
        print(data_file_path)
        data_list.append(args.dataset+ str(data_file_path))
        i +=1

    data[args.datasets_name] = data_list
    pprint.pprint(data,width=40)
    print("\n==================finished===================\n")
    if overwrite:
        with open(args.output_file,mode='wt', encoding='utf-8') as f:
            json.dump(data,f,indent=4)
            #config_file = json.load(f)

if __name__ == "__main__":
    main()
