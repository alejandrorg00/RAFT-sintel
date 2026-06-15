import sys
sys.path.append('core')

import argparse
import os
import numpy as np
import torch

import datasets
from raft import RAFT
from utils import frame_utils
from utils.utils import InputPadder, forward_interpolate
from flyvis_preprocessing.raft_hex_input import RAFTFlyVisHexInput


@torch.no_grad()
def create_sintel_submission(
    model,
    iters=32,
    warm_start=False,
    output_path='sintel_submission',
):
    """Create submission for the Sintel leaderboard."""
    model.eval()

    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(
            split='test',
            aug_params=None,
            dstype=dstype,
        )

        flow_prev, sequence_prev = None, None

        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]

            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(
                image1[None].cuda(),
                image2[None].cuda(),
            )

            flow_low, flow_pr = model(
                image1,
                image2,
                iters=iters,
                flow_init=flow_prev,
                test_mode=True,
            )

            flow = padder.unpad(flow_pr[0])
            flow = flow.permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(
                output_dir,
                'frame%04d.flo' % (frame + 1),
            )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence


@torch.no_grad()
def create_kitti_submission(
    model,
    iters=24,
    output_path='kitti_submission',
):
    """Create submission for the KITTI leaderboard."""
    model.eval()

    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(
            image1[None].cuda(),
            image2[None].cuda(),
        )

        _, flow_pr = model(
            image1,
            image2,
            iters=iters,
            test_mode=True,
        )

        flow = padder.unpad(flow_pr[0])
        flow = flow.permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)


@torch.no_grad()
def validate_chairs(model, iters=24):
    """Perform evaluation on the FlyingChairs validation split."""
    model.eval()

    epe_list = []
    val_dataset = datasets.FlyingChairs(split='validation')

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(
            image1,
            image2,
            iters=iters,
            test_mode=True,
        )

        epe = torch.sum((flow_pr[0].cpu() - flow_gt) ** 2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))

    print("Validation Chairs EPE: %f" % epe)

    return {'chairs': epe}


@torch.no_grad()
def validate_sintel(model, iters=32):
    """Perform validation using the full Sintel training split."""
    model.eval()

    results = {}

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(
            split='training',
            dstype=dstype,
        )

        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid = val_dataset[val_id]

            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(
                image1,
                image2,
                iters=iters,
                test_mode=True,
            )

            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            valid_mask = valid >= 0.5

            epe_list.append(
                epe[valid_mask].detach().cpu().view(-1).numpy()
            )

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print(
            "Validation Sintel (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f"
            % (dstype, epe, px1, px3, px5)
        )

        results[dstype] = epe
        results[f'val/{dstype}/epe'] = epe
        results[f'val/{dstype}/1px'] = px1
        results[f'val/{dstype}/3px'] = px3
        results[f'val/{dstype}/5px'] = px5

    return results


@torch.no_grad()
def validate_sintel_flyvis_split(
    model,
    iters=32,
    input_mode='rgb',
    flyvis_hex=False,
    tag='sintel_flyvis_split',
):
    """
    Validate on the FlyVis scene-level Sintel validation split.

    Modes:
        input_mode='rgb', flyvis_hex=False
            Regular RGB Sintel frames.

        input_mode='lum', flyvis_hex=False
            Luminance frames repeated across 3 channels.

        input_mode='rgb', flyvis_hex=True
            RGB frames passed through FlyVis-style hex preprocessing.

        input_mode='lum', flyvis_hex=True
            Luminance frames passed through FlyVis-style hex preprocessing.
    """
    model.eval()

    results = {}

    hex_preprocessor = None
    if flyvis_hex:
        hex_preprocessor = RAFTFlyVisHexInput(
            extent=15,
            kernel_size=13,
            output_size=256,
            device="cuda",
        )

    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(
            split='training',
            dstype=dstype,
            scenes=datasets.FLYVIS_VAL_SCENES,
            input_mode=input_mode,
        )

        epe_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, valid = val_dataset[val_id]

            image1 = image1[None].cuda(non_blocking=True)
            image2 = image2[None].cuda(non_blocking=True)
            flow_gt = flow_gt[None].cuda(non_blocking=True)
            valid = valid[None].cuda(non_blocking=True)

            if flyvis_hex:
                image1 = hex_preprocessor.batch_image_to_raft_input(
                    image1,
                    input_mode=input_mode,
                )
                image2 = hex_preprocessor.batch_image_to_raft_input(
                    image2,
                    input_mode=input_mode,
                )
                flow_gt = hex_preprocessor.batch_flow_to_raft_target(flow_gt)
                valid = hex_preprocessor.batch_valid_to_raft_mask(valid)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(
                image1,
                image2,
                iters=iters,
                test_mode=True,
            )

            flow = padder.unpad(flow_pr[0])

            flow_gt = flow_gt[0]
            valid = valid[0]

            epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
            valid_mask = valid >= 0.5

            epe_list.append(
                epe[valid_mask].detach().cpu().view(-1).numpy()
            )

        epe_all = np.concatenate(epe_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all < 1)
        px3 = np.mean(epe_all < 3)
        px5 = np.mean(epe_all < 5)

        print(
            "Validation %s (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f"
            % (tag, dstype, epe, px1, px3, px5)
        )

        results[f'val/{dstype}/epe'] = epe
        results[f'val/{dstype}/1px'] = px1
        results[f'val/{dstype}/3px'] = px3
        results[f'val/{dstype}/5px'] = px5

    return results


@torch.no_grad()
def validate_kitti(model, iters=24):
    """Perform validation using the KITTI-2015 training split."""
    model.eval()

    val_dataset = datasets.KITTI(split='training')

    out_list = []
    epe_list = []

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]

        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(
            image1,
            image2,
            iters=iters,
            test_mode=True,
        )

        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt) ** 2, dim=0).sqrt()
        mag = torch.sum(flow_gt ** 2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe / mag) > 0.05)).float()

        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))

    return {
        'kitti-epe': epe,
        'kitti-f1': f1,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")

    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument(
        '--mixed_precision',
        action='store_true',
        help='use mixed precision',
    )
    parser.add_argument(
        '--alternate_corr',
        action='store_true',
        help='use efficient correlation implementation',
    )
    parser.add_argument(
        '--iters',
        type=int,
        default=32,
        help='number of RAFT update iterations during evaluation',
    )

    args = parser.parse_args()

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))
    model.cuda()
    model.eval()

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, iters=args.iters)

        elif args.dataset in [
            'sintel_flyvis_split',
            'sintel_flyvis_split_rgb',
        ]:
            validate_sintel_flyvis_split(
                model.module,
                iters=args.iters,
                input_mode='rgb',
                flyvis_hex=False,
                tag='sintel_flyvis_split_rgb',
            )

        elif args.dataset == 'sintel_flyvis_split_lum':
            validate_sintel_flyvis_split(
                model.module,
                iters=args.iters,
                input_mode='lum',
                flyvis_hex=False,
                tag='sintel_flyvis_split_lum',
            )

        elif args.dataset == 'sintel_flyvis_split_hex_rgb':
            validate_sintel_flyvis_split(
                model.module,
                iters=args.iters,
                input_mode='rgb',
                flyvis_hex=True,
                tag='sintel_flyvis_split_hex_rgb',
            )

        elif args.dataset == 'sintel_flyvis_split_hex_lum':
            validate_sintel_flyvis_split(
                model.module,
                iters=args.iters,
                input_mode='lum',
                flyvis_hex=True,
                tag='sintel_flyvis_split_hex_lum',
            )

        elif args.dataset == 'kitti':
            validate_kitti(model.module, iters=args.iters)

        else:
            raise ValueError(f"Unknown evaluation dataset: {args.dataset}")