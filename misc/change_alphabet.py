#!/usr/bin/env python3
import argparse
import json
import os
from taiyaki.cmdargs import AutoBool, FileExists, FileAbsent
from taiyaki.common_cmdargs import add_common_command_args
from taiyaki.helpers import file_md5, load_model, open_file_or_stdout ,save_model
from taiyaki.json import JsonEncoder

parser = argparse.ArgumentParser(description='Dump JSON representation of model',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#add_common_command_args(parser, ["output"])

parser.add_argument('--params', default=True, action=AutoBool,
                    help='Output parameters as well as model structure')
parser.add_argument('--output',type=str,
                    help='Output directory')

parser.add_argument('--alphabet',type=str,
                    help='new alphabet, the canonical base must be set before modified ones')
parser.add_argument('model', action=FileExists, help='Model checkpoint')


def main():
    args = parser.parse_args()
    #model_md5 = file_md5(args.model)
    model = load_model(args.model)
    print("Previous alphabet",model.sublayers[-1].output_alphabet)

    for attr in ["mod_bases","mod_labels","mod_name_conv","ordered_mod_long_names"]:
        if "mod" in attr:
            print(attr,getattr(model.sublayers[-1],attr))
    #print("Previous alphabet",dir(model.sublayers[-1]))#).mod_long_names)

    model.sublayers[-1].output_alphabet = args.alphabet
    save_model(model,args.output)


if __name__ == "__main__":
    main()