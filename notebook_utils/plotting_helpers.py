"""Shared plotting helpers for ``analysis.ipynb``"""

import torch
from matplotlib.colors import hex2color


def plot_image_grid(images_data, fig, axes):
    for item in images_data:
        row = item["row"]
        col = item["col"]
        image = item["image"]
        title = item.get("title", "")
        cmap = item.get("cmap", "gray")

        if axes.ndim == 1:
            ax = axes[col]
        else:
            ax = axes[row, col]

        if title:
            ax.set_title(title, pad=0, fontsize=14)

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        imshow_kw = {"cmap": cmap}
        if "vmin" in item and "vmax" in item:
            imshow_kw["vmin"] = item["vmin"]
            imshow_kw["vmax"] = item["vmax"]
        ax.imshow(image, **imshow_kw)
        ax.axis("off")


def get_original_images(samples, labels, channels, start_row, title="Original Timepoint"):
    images_data = []
    for j, sample in enumerate(samples):
        images = sample.squeeze()

        if images.ndim == 2:
            images_data.append(
                {"image": images, "title": f"{title} {labels[j]}", "row": start_row, "col": j}
            )
        elif images.ndim == 3:
            for i, image in enumerate(images):
                t = f"{title} {labels[j]}" if i == 0 else ""
                images_data.append({"image": image, "title": t, "row": start_row + i, "col": j})
        else:
            raise ValueError("Unsupported image dimensions. Expected 2D or 3D images.")

    return images_data


def get_reconstructed_images(samples, model, channels, start_row, title="Predicted Timepoint"):
    images_data = []
    for j, sample in enumerate(samples):
        model.eval()
        sample_input = sample.unsqueeze(0)

        with torch.no_grad():
            result = model(sample_input)

        reconstructed_image, label_pred = result[:2]
        reconstructed_image = reconstructed_image.squeeze()
        label_pred = torch.max(label_pred, dim=1)[1].item()

        if reconstructed_image.ndim == 2:
            images_data.append(
                {"image": reconstructed_image, "title": f"{title}", "row": start_row, "col": j}
            )
        elif reconstructed_image.ndim == 3:
            for i, image in enumerate(reconstructed_image):
                t = f"{title} {label_pred}" if i == 0 else ""
                images_data.append({"image": image, "title": t, "row": start_row + i, "col": j})
        else:
            raise ValueError("Unsupported image dimensions. Expected 2D or 3D images.")

    return images_data


def get_difference_images(samples, model, channels, start_row, cmap="bwr"):
    imgs = []
    for j, sample in enumerate(samples):
        with torch.no_grad():
            recon, _ = model(sample.unsqueeze(0))[:2]
        recon, orig = recon.squeeze(), sample.squeeze()

        diffs = recon - orig
        if diffs.ndim == 2:
            diffs = diffs.unsqueeze(0)

        for ch_idx, diff in enumerate(diffs):
            m = diff.abs().max().item() or 1.0
            imgs.append(
                {
                    "image": diff,
                    "title": "",
                    "row": start_row + ch_idx,
                    "col": j,
                    "cmap": cmap,
                    "vmin": -m,
                    "vmax": m,
                }
            )
    return imgs


def get_saliency_maps(dataset, indices, model, start_row, title="Saliency Map"):
    images_data = []
    for j, idx in enumerate(indices):
        data = dataset[idx]
        input_tensor = data[0].unsqueeze(0)

        saliency_map = model.get_saliency_map(input_tensor).squeeze()

        images_data.append(
            {"image": saliency_map, "title": f"{title}", "row": start_row, "col": j, "cmap": "hot"}
        )

    return images_data


def get_shaded_color(base_hex, timepoint, min_tp, max_tp, light_factor=0.2):
    base_rgb = hex2color(base_hex)
    if max_tp == min_tp:
        normalized_tp = 1.0
    else:
        normalized_tp = (timepoint - min_tp) / (max_tp - min_tp)
    interpolation_amount = light_factor + normalized_tp * (1 - light_factor)

    shaded_rgb = [1.0 * (1 - interpolation_amount) + c * interpolation_amount for c in base_rgb]
    shaded_rgb = [max(0, min(1, c)) for c in shaded_rgb]
    return shaded_rgb
