import cv2
import os.path as osp
import logging
import argparse
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader
from models import create_model

#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, default='./options/test/LSRW-Huawei.yml', help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

def main():
    save_imgs = True
    model = create_model(opt)
    save_folder = './LSRW-Huawei/{}'.format(opt['name'])
    util.mkdirs(save_folder)
    
    print('mkdir finish')
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    
    for phase, dataset_opt in opt['datasets'].items():
        # 为每个数据集创建独立的子目录
        dataset_save_folder = osp.join(save_folder, phase)  # 使用 phase (Rain100L, Rain100H)
        
        GT_folder = osp.join(dataset_save_folder, 'images/GT')
        output_folder = osp.join(dataset_save_folder, 'images/output')
        input_folder = osp.join(dataset_save_folder, 'images/input')
        
        util.mkdirs(GT_folder)
        util.mkdirs(output_folder)
        util.mkdirs(input_folder)
        
        print(f'Processing dataset: {phase}')
        print(f'Saving to: {dataset_save_folder}')
        
        val_set = create_dataset(dataset_opt)
        val_loader = create_dataloader(val_set, dataset_opt, opt, None)
        
        for val_data in val_loader:
            model.feed_data(val_data)
            model.test()
            visuals = model.get_current_visuals()
            rlt_img = util.tensor2img(visuals['rlt'])
            gt_img = util.tensor2img(visuals['GT'])
            input_img = util.tensor2img(visuals['LQ'])
            
            if save_imgs:
                try:
                    # 获取原始文件名（不包含路径）
                    # 假设 val_data['folder'] 或 val_data['LQ_path'] 包含文件名信息
                    if 'LQ_path' in val_data:
                        filename = osp.basename(val_data['LQ_path'])
                    elif 'folder' in val_data:
                        # 原来的提取方式，但只取文件名
                        tag = str(val_data['folder']).split('[')[1].split(']')[0]
                        tag = tag.replace("'", "")
                        filename = tag
                    else:
                        # 使用索引作为文件名
                        filename = f"image_{idx}.png"
                    
                    # 保存到对应的数据集目录
                    cv2.imwrite(osp.join(output_folder, filename), rlt_img)
                    cv2.imwrite(osp.join(GT_folder, filename), gt_img)
                    cv2.imwrite(osp.join(input_folder, filename), input_img)
                    
                    print(f'Saved: {phase}/{filename}')
                    
                except Exception as e:
                    print(f'Error saving {phase}: {e}')
                    import ipdb; ipdb.set_trace()

if __name__ == '__main__':
    main()
