import pandas as pd 
import numpy as np

def filter_roi_table(nwbfile) -> pd.DataFrame:
    """
    Parameters:
    nwbfile: NWBFile 
        NWBFile object containing roi table with image segmentation masks and soma classifications 

    Outputs:
    filtered_roi_table: pd.DataFrame
        filtered ROI table with only ROIs that pass soma classifier and photostimulated and conditioned neurons
    """
    # Get photostim and cn indices from tables in NWB file
    photostim_ids = nwbfile.stimulus["PhotostimTrials"].to_dataframe().closest_roi.values
    cn_id = nwbfile.stimulus["Trials"].to_dataframe().closest_roi.values
    roi_table = nwbfile.processing["processed"].data_interfaces["image_segmentation"].plane_segmentations["roi_table"][:]

    missing_ids = np.unique(np.concatenate([photostim_ids, cn_id]))

    # Check which indices did not pass soma detection
    add_ids = []
    for idx in missing_ids:
        if roi_table.iloc[idx].is_soma == 0:
            add_ids.append(idx)
        else:
            continue

    # Filter roi table for those that pass is_soma and for indices of missing ids
    filtered_roi_table = roi_table[roi_table.is_soma==1]
    filtered_roi_table = pd.concat(
    [filtered_roi_table, roi_table.iloc[add_ids]],
    ignore_index=False  # reindex rows if you don’t care about preserving original indices
    )

    filtered_roi_table = filtered_roi_table[['image_mask']]

    return filtered_roi_table

def filter_dff(nwbfile, filtered_roi_table) -> pd.DataFrame:
    """
    Gets dff traces for ROIs that are likely somas and removes columns that are all NaNs.

    Parameters:
    nwbfile : NWBFile
        NWBFile object containing processed dff traces.
    filtered_roi_table : pd.DataFrame
        Filtered ROI table with only ROIs that pass soma classifier

    Outputs: 
    cleaned_dff : np.ndarray
        Filtered array (n_frames x n_ROIs) with NaN-only columns removed.
    cleaned_roi_table : pd.DataFrame
        Filtered ROI table with corresponding ROIs removed.
    """

    # Load dff traces
    dff_traces = nwbfile.processing["processed"].data_interfaces["dff"].roi_response_series["dff"].data[:]

    # Select ROIs of interest
    filtered_dff_traces = dff_traces[:, filtered_roi_table.index]

    # Identify columns that are all NaN
    mask = ~np.all(np.isnan(filtered_dff_traces), axis=0)

    # Filter out NaN-only columns
    cleaned_dff = filtered_dff_traces[:, mask]
    cleaned_roi_table = filtered_roi_table[mask]

    return cleaned_dff, cleaned_roi_table

        