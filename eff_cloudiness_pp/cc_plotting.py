import geopandas as gpd
from shapely.geometry import box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_single_field(parameter, str_par, time, template, vmin=None, vmax=None):
    str_time = time.strftime("%Y%m%d%H%M")
    plt.figure(figsize=(10, 6))
    x = template["x"].values
    y = template["y"].values
    if vmin is not None and vmax is not None:
        levels = np.linspace(vmin, vmax, 25)  # 100 contour levels
    else:
        levels = 10      
    contour = plt.contourf(x, y, parameter, levels=levels, cmap="viridis", extend="both")
    cbar = plt.colorbar(contour, label=str_par)
    # Set ticks only if vmin and vmax are provided
    if vmin is not None and vmax is not None:
        ticks = np.linspace(vmin, vmax, 5)
        cbar.set_ticks(ticks)
    #cbar.set_ticks(np.linspace(vmin, vmax, 5) if vmin is not None and vmax is not None else None)
    plt.title(str_par)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("{}_{}.png".format(str_par, str_time))
    plt.close()

def plot_single_border(parameter, lats, lons, str_par, time, vmin=None, vmax=None, minlat=None, maxlat=None, minlon=None, maxlon=None):
    str_time = time.strftime("%Y%m%d%H%M")
    plt.figure(figsize=(10, 6))
    shapefile_path = "/home/users/hietal/natural_earth/ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)

    # Define the bounding box for your data
    if (minlat is not None and maxlat is not None and minlon is not None and maxlon is not None):
        bbox = box(minlon, minlat, maxlon, maxlat)
    else:
        bbox = box(lons.min(), lats.min(), lons.max(), lats.max())
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")

    # Clip the world data to the bounding box
    world_clipped = gpd.clip(world, bbox_gdf)

    # Correct the orientation if needed
    if parameter.shape != lats.shape:
        parameter = parameter.T  # Fix transposition
    if lats.shape != lons.shape:
        lons, lats = np.meshgrid(lons, lats)

    # Plot data
    plt.contourf(lons, lats, parameter, levels=np.linspace(vmin, vmax, 25), cmap="viridis", origin="lower")

    if (minlat is not None and maxlat is not None and minlon is not None and maxlon is not None):
        plt.xlim(minlon, maxlon)
        plt.ylim(minlat, maxlat)
    # Overlay clipped land borders
    world_clipped.boundary.plot(ax=plt.gca(), color="black", linewidth=0.5)

    # Add labels and colorbar
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"{str_par} - {str_time}")
    plt.colorbar(label=str_par)

    # Save the plot
    plt.savefig(f"{str_par}_{str_time}.png", bbox_inches="tight")
    plt.close()

def plot_two_border(
    parameter1, parameter2, lats, lons, str_par1, str_par2, time, 
    vmin=None, vmax=None, minlat=None, maxlat=None, minlon=None, maxlon=None
):
    str_time = time.strftime("%Y%m%d%H%M")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Create two side-by-side subplots
    shapefile_path = "/home/users/hietal/natural_earth/ne_110m_admin_0_countries.shp"
    world = gpd.read_file(shapefile_path)

    # Define the bounding box for your data
    if (minlat is not None and maxlat is not None and minlon is not None and maxlon is not None):
        bbox = box(minlon, minlat, maxlon, maxlat)
    else:
        bbox = box(lons.min(), lats.min(), lons.max(), lats.max())
    bbox_gdf = gpd.GeoDataFrame({"geometry": [bbox]}, crs="EPSG:4326")

    # Clip the world data to the bounding box
    world_clipped = gpd.clip(world, bbox_gdf)

    # Ensure proper orientation
    if parameter1.shape != lats.shape:
        parameter1 = parameter1.T
    if parameter2.shape != lats.shape:
        parameter2 = parameter2.T
    if lats.shape != lons.shape:
        lons, lats = np.meshgrid(lons, lats)

    # Plot the first field
    ax1 = axes[0]
    contour1 = ax1.contourf(
        lons, lats, parameter1, levels=np.linspace(vmin, vmax, 25), cmap="viridis", origin="lower"
    )
    if (minlat is not None and maxlat is not None and minlon is not None and maxlon is not None):
        ax1.set_xlim(minlon, maxlon)
        ax1.set_ylim(minlat, maxlat)
    world_clipped.boundary.plot(ax=ax1, color="black", linewidth=0.5)
    ax1.set_title(f"{str_par1} - {str_time}")
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    fig.colorbar(contour1, ax=ax1, label=str_par1)

    # Plot the second field
    ax2 = axes[1]
    contour2 = ax2.contourf(
        lons, lats, parameter2, levels=np.linspace(vmin, vmax, 25), cmap="viridis", origin="lower"
    )
    if (minlat is not None and maxlat is not None and minlon is not None and maxlon is not None):
        ax2.set_xlim(minlon, maxlon)
        ax2.set_ylim(minlat, maxlat)
    world_clipped.boundary.plot(ax=ax2, color="black", linewidth=0.5)
    ax2.set_title(f"{str_par2} - {str_time}")
    ax2.set_xlabel("Longitude")
    ax2.set_ylabel("Latitude")
    fig.colorbar(contour2, ax=ax2, label=str_par2)

    # Adjust layout and save the plot
    #plt.tight_layout()
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(f"{str_par1}_{str_par2}_{str_time}.png", bbox_inches="tight")
    plt.close()