"""
3D Gaussian Splatting Converter
Copyright (c) 2023 Francesco Fugazzi

This software is released under the MIT License.
For more information about the license, please see the LICENSE file.
"""

import numpy as np
from .base_converter import BaseConverter
from .utility import Utility
from .utility_functions import debug_print
from . import config

class FormatCC(BaseConverter):
    def to_3dgs(self):
        debug_print("[DEBUG] Starting conversion from CC to 3DGS...")

        # Load vertices from the updated data after all filters
        vertices = self.data
        debug_print(f"[DEBUG] Loaded {len(vertices)} vertices.")

        # Create a new structured numpy array for 3DGS format
        dtype_3dgs = self.define_dtype(has_scal=False, has_rgb=False)  # Define 3DGS dtype without any prefix
        converted_data = np.zeros(vertices.shape, dtype=dtype_3dgs)

        # Use the helper function to copy the data from vertices to converted_data
        Utility.copy_data_with_prefix_check(vertices, converted_data, ["", "scal_", "scalar_", "scalar_scal_"])

        debug_print("[DEBUG] Data copying completed.")
        debug_print("[DEBUG] Sample of converted data (first 5 rows):")
        if config.DEBUG:
            for i in range(5):
                debug_print(converted_data[i])

        debug_print("[DEBUG] Conversion from CC to 3DGS completed.")
        return converted_data


    def to_cc(self, process_rgb=False):
        debug_print("[DEBUG] Processing CC data...")

        # Check if RGB processing is required
        if process_rgb and not self.has_rgb():
            self.add_rgb()
            debug_print("[DEBUG] RGB added to data.")
        else:
            debug_print("[DEBUG] RGB processing is skipped or data already has RGB.")
        
        converted_data = self.data
        
        # For now, we'll just return the converted_data for the sake of this integration
        debug_print("[DEBUG] CC data processing completed.")
        return converted_data

    def add_or_ignore_rgb(self, process_rgb=True):
        debug_print("[DEBUG] Checking RGB for CC data...")

        # If RGB processing is required and RGB is not present
        if process_rgb and not self.has_rgb():
            # Compute RGB values for the data
            rgb_values = Utility.compute_rgb_from_vertex(self.data)

            # Get the new dtype definition from the BaseConverter class
            new_dtype_list, _ = BaseConverter.define_dtype(has_scal=True, has_rgb=True)
            new_dtype = np.dtype(new_dtype_list)

            # Create a new structured array that includes fields for RGB
            # It should have the same number of rows as the original data
            converted_data = np.zeros(self.data.shape[0], dtype=new_dtype)

            # Copy the data to the new numpy array, preserving existing fields
            for name in self.data.dtype.names:
                converted_data[name] = self.data[name]

            # Add the RGB values to the new numpy array
            converted_data['red'] = rgb_values[:, 0]
            converted_data['green'] = rgb_values[:, 1]
            converted_data['blue'] = rgb_values[:, 2]

            self.data = converted_data  # Update the instance's data with the new data
            debug_print("[DEBUG] RGB added to data.")
        else:
            debug_print("[DEBUG] RGB processing is skipped or data already has RGB.")
            converted_data = self.data  # If RGB is not added or skipped, the converted_data is just the original data.

        # Return the converted_data
        debug_print("[DEBUG] RGB check for CC data completed.")
        return converted_data