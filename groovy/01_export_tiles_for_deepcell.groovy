/**
 * 01_export_tiles_for_deepcell.groovy
 * 
 * Deep Learning Cell Segmentation Pipeline
 * 
 * Purpose:
 *   Exports multichannel image tiles (Nuclear + Membrane/Cytoplasm channels)
 *   from Whole Slide Images (WSI) or Selected Annotations in QuPath for
 *   consumption by DeepCell (Mesmer) segmentation.
 *
 * Requirements:
 *   - QuPath 0.4.x or 0.5.x
 *
 * Usage:
 *   1. Open a multiplexed image or WSI in QuPath.
 *   2. (Optional) Draw annotations around regions of interest (ROIs).
 *      If no annotations exist, the full valid image area is tiled.
 *   3. Adjust target channel indices / names and output directory below.
 *   4. Run this script in QuPath (Automate -> Script Editor -> Run).
 */

import qupath.lib.images.ImageData
import qupath.lib.images.servers.ImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.regions.ImageRegion
import qupath.lib.roi.interfaces.ROI
import qupath.lib.objects.PathAnnotationObject
import qupath.lib.images.writers.ImageWriterTools
import groovy.json.JsonBuilder
import java.awt.image.BufferedImage
import java.io.File

// ==========================================
// CONFIGURATION
// ==========================================
def outputDirBase = "c:/Users/micke/OneDrive/Documents/DEEPCELL/data/tiles_export"
def targetTileSize = 1024        // Tile width and height in pixels
def overlap = 64                 // Overlap between adjacent tiles in pixels
def targetMpp = 0.5              // Target resolution (DeepCell Mesmer expects ~0.5 um/pixel)

// Specify 0-indexed channel numbers or set channel names to search dynamically
// E.g., Nuclear channel (DAPI/Hoechst) and Membrane/Cytoplasm (PanCK/CD45/NaKATPase)
def nuclearChannelIndex = 0
def membraneChannelIndex = 1

// ==========================================
// INITIALIZATION
// ==========================================
def imageData = getCurrentImageData()
if (imageData == null) {
    println "[ERROR] No image open in QuPath. Please open an image before running."
    return
}

def server = imageData.getServer()
def imageName = server.getMetadata().getName().replaceAll("[^a-zA-Z0-9._-]", "_")
def outputDir = new File(outputDirBase, imageName)
if (!outputDir.exists()) {
    outputDir.mkdirs()
}

def cal = server.getPixelCalibration()
double currentMpp = cal.hasPixelSizeMicrons() ? (cal.getPixelWidthMicrons() + cal.getPixelHeightMicrons()) / 2.0 : 0.5
double downsample = targetMpp / currentMpp
if (downsample < 1.0) downsample = 1.0

println "=========================================================="
println "Exporting DeepCell Tiles for Image: ${imageName}"
println "Current Resolution: ${currentMpp} um/px | Target: ${targetMpp} um/px (Downsample: ${downsample})"
println "Tile Size: ${targetTileSize}x${targetTileSize} (Overlap: ${overlap} px)"
println "Output Directory: ${outputDir.getAbsolutePath()}"
println "=========================================================="

// Determine areas to export: selected annotations or full image
def hierarchy = imageData.getHierarchy()
def annotations = hierarchy.getAnnotationObjects().findAll { it.getROI() != null }

def regionsToProcess = []
if (annotations.isEmpty()) {
    println "[INFO] No annotations found. Tiling full image dimensions: ${server.getWidth()} x ${server.getHeight()}"
    regionsToProcess << [
        x: 0,
        y: 0,
        width: server.getWidth(),
        height: server.getHeight(),
        name: "Full_Image"
    ]
} else {
    println "[INFO] Found ${annotations.size()} annotations to tile."
    annotations.eachWithIndex { annot, idx ->
        def roi = annot.getROI()
        regionsToProcess << [
            x: (int) Math.floor(roi.getBoundsX()),
            y: (int) Math.floor(roi.getBoundsY()),
            width: (int) Math.ceil(roi.getBoundsWidth()),
            height: (int) Math.ceil(roi.getBoundsHeight()),
            name: "Annotation_${idx + 1}"
        ]
    }
}

// ==========================================
// TILING LOOP & EXPORT
// ==========================================
def tileMetadata = []
int tileCount = 0
int step = targetTileSize - overlap

regionsToProcess.each { reg ->
    int startX = reg.x
    int startY = reg.y
    int endX = reg.x + reg.width
    int endY = reg.y + reg.height

    for (int y = startY; y < endY; y += step) {
        for (int x = startX; x < endX; x += step) {
            int w = Math.min(targetTileSize, server.getWidth() - x)
            int h = Math.min(targetTileSize, server.getHeight() - y)

            if (w <= 0 || h <= 0) continue

            // Zero-padded filename
            String tileId = String.format("%s_tile_%05d_x%d_y%d", imageName, tileCount, x, y)
            File tileFile = new File(outputDir, "${tileId}.tif")

            // Create region request for the channels
            // Request channels as array [nuclearChannelIndex, membraneChannelIndex]
            def channelsToExport = [nuclearChannelIndex, membraneChannelIndex]
            
            // Build RegionRequest
            def request = RegionRequest.createInstance(server.getPath(), downsample, x, y, w, h)
            
            try {
                // Write multichannel image tile using QuPath's ImageWriterTools
                ImageWriterTools.writeImageRegion(server, request, tileFile.getAbsolutePath())
                
                tileMetadata << [
                    tile_id: tileId,
                    file_name: tileFile.getName(),
                    file_path: tileFile.getAbsolutePath(),
                    region_parent: reg.name,
                    x: x,
                    y: y,
                    width: w,
                    height: h,
                    downsample: downsample,
                    nuclear_channel: nuclearChannelIndex,
                    membrane_channel: membraneChannelIndex
                ]
                tileCount++
            } catch (Exception e) {
                println "[WARN] Failed to export tile at (${x}, ${y}): ${e.getMessage()}"
            }
        }
    }
}

// Save Metadata JSON for Python pipeline reconstruction
File metaFile = new File(outputDir, "tiles_metadata.json")
metaFile.text = new JsonBuilder([
    image_name: imageName,
    pixel_size_um: currentMpp,
    target_pixel_size_um: targetMpp,
    downsample: downsample,
    total_tiles: tileCount,
    tiles: tileMetadata
]).toPrettyString()

println "[SUCCESS] Finished exporting ${tileCount} tiles."
println "[INFO] Metadata catalog written to: ${metaFile.getAbsolutePath()}"

