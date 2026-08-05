import os
import argparse
from options import options as option
from data import create_dataset, create_dataloader
from models import create_model


def main(opt_path, ckpt_path, num_images=5, dataroot_LQ='', dataroot_GT='', save_base_dir=''):
    # Parse options
    opt = option.parse(opt_path, is_train=False)
    # override pretrained model path
    if ckpt_path:
        if 'path' not in opt:
            opt['path'] = {}
        opt['path']['pretrain_model_G'] = ckpt_path
    # override dataset roots if provided
    if dataroot_LQ or dataroot_GT:
        for phase, ds_opt in opt.get('datasets', {}).items():
            if dataroot_LQ:
                ds_opt['dataroot_LQ'] = dataroot_LQ
            if dataroot_GT:
                ds_opt['dataroot_GT'] = dataroot_GT
    opt = option.dict_to_nonedict(opt)
    if save_base_dir:
        opt['save_base_dir'] = save_base_dir

    # create model
    model = create_model(opt)

    # attempt to set save_intermediate and save_base_dir on underlying netG
    try:
        netG = model.netG
        # if DataParallel
        if hasattr(netG, 'module'):
            net = netG.module
        else:
            net = netG
        # enable saving
        if hasattr(net, 'save_intermediate'):
            net.save_intermediate = True
        if save_base_dir:
            if hasattr(net, 'save_base_dir'):
                net.save_base_dir = save_base_dir
        if hasattr(net, 'save_base_dir'):
            base = net.save_base_dir
            os.makedirs(base, exist_ok=True)
        print(f"Enabled intermediate saving at: {getattr(net, 'save_base_dir', './intermediates_cwnet')}")
    except Exception as e:
        print('Warning: could not set save_intermediate on model:', e)

    # create dataset and dataloader
    datasets = opt.get('datasets', {})
    # try to find a phase dataset (val/test)
    chosen = None
    for phase, ds_opt in datasets.items():
        chosen = (phase, ds_opt)
        break
    if chosen is None:
        raise RuntimeError('No datasets found in options')
    phase, ds_opt = chosen
    dataset = create_dataset(ds_opt)
    dataloader = create_dataloader(dataset, ds_opt, opt, None)

    # iterate and run
    cnt = 0
    for i, data in enumerate(dataloader):
        if num_images and cnt >= num_images:
            break
        model.feed_data(data)
        model.test()
        visuals = model.get_current_visuals(need_GT=('GT' in data))
        print(f'Processed image {cnt+1}')
        cnt += 1

    print('Done smoke test. Saved intermediates under model.save_base_dir (timestamped subfolders).')

    # --- Aggregate DWT metrics for the latest run (per-level, per-step) ---
    try:
        import glob, csv, statistics
        base = getattr(net, 'save_base_dir', None) or save_base_dir or './intermediates_cwnet'
        # find latest run directory
        runs = [d for d in glob.glob(os.path.join(base, '*')) if os.path.isdir(d)]
        if runs:
            latest = sorted(runs, key=os.path.getmtime, reverse=True)[0]
            print('Aggregating metrics in latest run:', latest)
            # collect metrics from metrics_*.csv files under latest
            metrics_files = glob.glob(os.path.join(latest, '**', 'metrics_*.csv'), recursive=True)
            # organize: data[level][step]['lf'/'hf'] -> list
            from collections import defaultdict
            data = defaultdict(lambda: defaultdict(lambda: {'lf': [], 'hf': []}))
            for mf in metrics_files:
                try:
                    with open(mf) as fh:
                        reader = csv.reader(fh)
                        hdr = next(reader)
                        for row in reader:
                            if not row: continue
                            level = row[0]
                            step = int(row[1])
                            metric = row[2]
                            val = float(row[3])
                            if metric == 'dwt_lf_ratio':
                                data[level][step]['lf'].append(val)
                            elif metric == 'dwt_hf_ratio':
                                data[level][step]['hf'].append(val)
                except Exception:
                    pass
            levels = ['conv1','conv2','conv3','conv4','conv5']
            summary_csv = os.path.join(latest, 'summary_conv1-5_steps0-2.csv')
            with open(summary_csv, 'w', newline='') as sf:
                writer = csv.writer(sf)
                writer.writerow(['level','step','lf_mean','lf_sd','lf_n','hf_mean','hf_sd','hf_n'])
                for lvl in levels:
                    for step in [0,1,2]:
                        lfs = data[lvl][step]['lf']
                        hfs = data[lvl][step]['hf']
                        if lfs:
                            lf_mean = sum(lfs)/len(lfs)
                            lf_sd = statistics.pstdev(lfs) if len(lfs)>1 else 0.0
                            lf_n = len(lfs)
                        else:
                            lf_mean = lf_sd = None; lf_n = 0
                        if hfs:
                            hf_mean = sum(hfs)/len(hfs)
                            hf_sd = statistics.pstdev(hfs) if len(hfs)>1 else 0.0
                            hf_n = len(hfs)
                        else:
                            hf_mean = hf_sd = None; hf_n = 0
                        writer.writerow([lvl, step, lf_mean, lf_sd, lf_n, hf_mean, hf_sd, hf_n])
            print('Wrote aggregated summary:', summary_csv)
        else:
            print('No run directories found under', base)
    except Exception as e:
        print('Aggregation failed:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default='')
    parser.add_argument('--num_images', type=int, default=5)
    parser.add_argument('--dataroot_LQ', type=str, default='')
    parser.add_argument('--dataroot_GT', type=str, default='')
    parser.add_argument('--save_base_dir', type=str, default='')
    args = parser.parse_args()
    main(args.opt, args.ckpt, args.num_images, args.dataroot_LQ, args.dataroot_GT, args.save_base_dir)
