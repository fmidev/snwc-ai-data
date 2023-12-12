import numpy as np
import matplotlib.pyplot as plt

def plot(obs, background, output, diff, lons, lats, args):
    vmin1 = -5
    vmax1 = 5
    if args.parameter == "temperature":
        obs_parameter = "TA_PT1M_AVG"
        output = list(map(lambda x: x - 273.15, output))
    elif args.parameter == "windspeed":
        obs_parameter = "WSP_PT10M_AVG"
    elif args.parameter == "gust":
        obs_parameter = "WG_PT1H_MAX"
    elif args.parameter == "humidity":
        obs_parameter = "RH_PT1M_AVG"
        output = np.multiply(output, 100)
        vmin1 = -30
        vmax1 = 30

    vmin = min(np.amin(background), np.amin(output))
    vmax = min(np.amax(background), np.amax(output))

    # vmin1 =  np.amin(diff)
    # vmax1 =  np.amax(diff)

    for k in range(0, len(diff)):
        plt.figure(figsize=(13, 6), dpi=80)

        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            background[k + 1],
            cmap="Spectral_r",  # "RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="MNWC " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        plt.subplot(1, 3, 2)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            diff[k],
            cmap="RdBu_r",
            vmin=vmin1,
            vmax=vmax1,
        )

        """
        plt.scatter(
        obs["longitude"],
        obs["latitude"],
        s=10,
        c=obs[obs_parameter],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        )
        """
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        plt.subplot(1, 3, 3)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            output[k],
            cmap="Spectral_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="XGB " + str(k) + "h " + args.parameter, orientation="horizontal"
        )

        # plt.show()
        plt.savefig("all_" + args.parameter + str(k) + ".png")
