import matplotlib.pyplot as plt
import seaborn as sns


def plot_first_batch(batch):
    """
    batch contains:
    pitch, energy, vad, pitch_masked, energy_masked, vad_masked, mask_pad, mask_pred
    """
    batch_size = len(batch["pitch"])

    fig, axs = plt.subplots(8, batch_size, figsize=(batch_size * 2, 10))
    for i in range(batch_size):
        sns.lineplot(
            x=range(len(batch["pitch"][i])),
            y=batch["pitch"][i],
            ax=axs[0, i],
        )
        sns.lineplot(
            x=range(len(batch["energy"][i])),
            y=batch["energy"][i],
            ax=axs[1, i],
        )
        sns.lineplot(
            x=range(len(batch["vad"][i])),
            y=batch["vad"][i],
            ax=axs[2, i],
        )
        sns.lineplot(
            x=range(len(batch["pitch_masked"][i])),
            y=batch["pitch_masked"][i],
            ax=axs[3, i],
        )
        sns.lineplot(
            x=range(len(batch["energy_masked"][i])),
            y=batch["energy_masked"][i],
            ax=axs[4, i],
        )
        sns.lineplot(
            x=range(len(batch["vad_masked"][i])),
            y=batch["vad_masked"][i],
            ax=axs[5, i],
        )
        sns.lineplot(
            x=range(len(batch["mask_pad"][i])),
            y=batch["mask_pad"][i],
            ax=axs[6, i],
        )
        sns.lineplot(
            x=range(len(batch["mask_pred"][i])),
            y=batch["mask_pred"][i],
            ax=axs[7, i],
        )
    return fig
