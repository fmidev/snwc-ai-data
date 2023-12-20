import numpy as np
import matplotlib.pyplot as plt
import gridpp
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
import datetime

def density_scatter( x , y, ax = None, sort = True, bins = 20, yla = '', xla = '', picname='scatterplot',  **kwargs)   :
	"""
	code originally from
	https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density/53865762#53865762
	Scatter plot colored by 2d histogram
	"""
	if ax is None :
		fig , ax = plt.subplots()
	data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
	z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

	#To be sure to plot all data
	z[np.where(np.isnan(z))] = 0.0

	# Sort the points by density, so that the densest points are plotted last
	if sort :
		idx = z.argsort()
		x, y, z = x[idx], y[idx], z[idx]

	ax.scatter( x, y, c=z, **kwargs )
	lineStart = x.min()
	lineEnd = x.max()
	#add diagonal
	ax.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-', color = 'r')
	plt.xlabel(xla)
	plt.ylabel(yla)

	norm = Normalize(vmin = np.min(z), vmax = np.max(z))
	cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
	cbar.ax.set_ylabel('Density')
	plt.savefig(picname + '.png')
	return ax

def plot(grid, points, obs, background, output, diff, lons, lats, param, analysistime):
    timestamp = analysistime.strftime("%y%m%d") 
    #print("timestamp:", timestamp)
    vmin1 = -5
    vmax1 = 5
    if param == "t":
        output = list(map(lambda x: x - 273.15, output))
    elif param == "r":
        #obs_parameter = "RH_PT1M_AVG"
        output = np.multiply(output, 100)
        vmin1 = -30
        vmax1 = 30

    vmin = min(np.amin(background), np.amin(output))
    vmax = min(np.amax(background), np.amax(output))

    # vmin1 =  np.amin(diff)
    # vmax1 =  np.amax(diff)

    for k in range(0, len(diff)):
        one_bg = gridpp.nearest(grid, points[k], background[k]) # raw model vals in obs points
        orig_diff = gridpp.nearest(grid, points[k], diff[k]) # diff field in obs points
        one_out = gridpp.nearest(grid, points[k], output[k]) # corrected vals in obs points
        tmp_obs = obs[k]
        one_obs = tmp_obs['obs_value'].values
        diff_points = one_out - one_obs # calculated from output data
        #diff_points2 = (one_bg - orig_diff) - one_obs # WS diff field in obs points
        #if args.parameter == "uv":
        #    diff_points = diff_points2 
        plt.figure(figsize=(13, 6), dpi=80)
        plt.subplot(1, 3, 1)
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            background[k],
            cmap="Spectral_r",  # "RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )

        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="MEPS " + str(k) + "h " + param, orientation="horizontal"
        )
          
        plt.subplot(1, 3, 2)
        """
        # plot difference in grid  
        plt.pcolormesh(
            np.asarray(lons),
            np.asarray(lats),
            diff[k],
            cmap="RdBu_r",
            vmin=vmin1,
            vmax=vmax1,
        )
        """
        # plot difference in points
        plt.scatter(
            tmp_obs["longitude"],
            tmp_obs["latitude"],
            s=10,
            c=diff_points,
            cmap="RdBu_r",
            vmin=vmin1,
            vmax=vmax1,
        )
        #"""
        """
        # plot the true observations
        plt.scatter(
        obso["longitude"],
        obso["latitude"],
        s=10,
        c=obso['obs_value'],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        )
        """
        plt.xlim(0, 35)
        plt.ylim(55, 75)
        cbar = plt.colorbar(
            label="Diff " + str(k) + "h " + param, orientation="horizontal"
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
            label="Analysis " + str(k) + "h " + param, orientation="horizontal"
        )

        # plt.show()
        plt.savefig(timestamp + "all_" + param + str(k) + ".png")

def plot_scatter(grid, points, obs, background, output, param, analysistime):
    timestamp = analysistime.strftime("%y%m%d")
    vmin1 = -5 # np.amin(diff)
    vmax1 = 5 # np.amax(diff)
    if param == "t":
        output = list(map(lambda x: x - 273.15, output))
    elif param == "r":
        output = np.multiply(output, 100)
        vmin1 = -30
        vmax1 = 30
    vmin = min(np.amin(background), np.amin(output))
    vmax = min(np.amax(background), np.amax(output))

    for i in range(0, len(output)):
        one_bg = gridpp.nearest(grid, points[i], background[i]) # raw model vals in obs points
        one_out = gridpp.nearest(grid, points[i], output[i]) # corrected vals in obs points
        tmp_obs = obs[i]
        one_obs = tmp_obs['obs_value'].values

        density_scatter(one_obs, one_bg, bins = [30,30],yla = 'MEPS',xla='Observations',picname = timestamp + "_MEPS_" + str(i) + "h" + param)
        density_scatter(one_obs, one_out, bins = [30,30],yla ='Analysis',xla='Observations',picname = timestamp + "_Analysis_" + str(i) + "h" + param)

#        # plt.show()
#        plt.savefig("all_" + args.parameter + str(k) + ".png")
#        plot_scatter()
