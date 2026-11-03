/**
 * 03_batch_project_pipeline.groovy
 * 
 * Deep Learning Cell Segmentation & Spatial Biology Pipeline
 * 
 * Purpose:
 *   High-Throughput Batch Automation for QuPath Projects.
 *   Iterates through all images in an open or headless QuPath project,
 *   extracts tissue annotations, normalizes pixel resolutions, exports tiles,
 *   and writes a master project manifest for the DeepCell Python pipeline.
 *
 * Requirements:
 *   - QuPath 0.4.x or 0.5.x
 *
 * Usage (GUI):
 *   1. Open a QuPath Project containing multiplexed images.
 *   2. Run this script in Automate -> Project -> Run for project.
 *
 * Usage (Headless CLI):
 *   QuPath script 03_batch_project_pipeline.groovy -p /path/to/project.qpproj
 */

import qupath.lib.images.ImageData
import qupath.lib.images.servers.ImageServer
import qupath.lib.regions.RegionRequest
import qupath.lib.images.writers.ImageWriterTools
import groovy.json.JsonBuilder
import java.io.File

// ==========================================
// CONFIGURATION
// ==========================================
def outputDirBase = "c:/Users/micke/OneDrive/Documents/DEEPCELL/data/project_tiles_export"
def targetTileSize = 1024
def overlap = 64
def targetMpp = 0.5

def nuclearChannelIndex = 0
def membraneChannelIndex = 1

// ==========================================
// BATCH EXECUTION
// ==========================================
def project = getProject()
if (project == null) {
    println "[ERROR] No QuPath project open. Please open a project or pass -p <project.qpproj>."
    return
}

def imageEntries = project.getImageList()
println "=========================================================="
println "BATCH QUPATH PIPELINE: Processing ${imageEntries.size()} Images"
println "Target Resolution: ${targetMpp} um/px | Tile Size: ${targetTileSize}x${targetTileSize}"
println "Master Export Root: ${outputDirBase}"
println "=========================================================="

def masterManifest = [:]

imageEntries.eachWithIndex { entry, imgIdx ->
    def imageData = entry.readImageData()
    def server = imageData.getServer()
    def imageName = server.getMetadata().getName().replaceAll("[^a-zA-Z0-9._-]", "_")
    
    println "\n[${imgIdx + 1}/${imageEntries.size()}] Initializing Image: ${imageName}"
    
    File imgOutputDir = new File(outputDirBase, imageName)
    if (!imgOutputDir.exists()) imgOutputDir.mkdirs()

    def cal = server.getPixelCalibration()
    double currentMpp = cal.hasPixelSizeMicrons() ? (cal.getPixelWidthMicrons() + cal.getPixelHeightMicrons()) / 2.0 : 0.5
    double downsample = targetMpp / currentMpp
    if (downsample < 1.0) downsample = 1.0

    int step = targetTileSize - overlap
    int tileCount = 0
    def tileList = []

    for (int y = 0; y < server.getHeight(); y += step) {
        for (int x = 0; x < server.getWidth(); x += step) {
            int w = Math.min(targetTileSize, server.getWidth() - x)
            int h = Math.min(targetTileSize, server.getHeight() - y)
            if (w <= 0 || h <= 0) continue

            String tileId = String.format("%s_tile_%05d_x%d_y%d", imageName, tileCount, x, y)
            File tileFile = new File(imgOutputDir, "${tileId}.tif")
            def request = RegionRequest.createInstance(server.getPath(), downsample, x, y, w, h)

            try {
                ImageWriterTools.writeImageRegion(server, request, tileFile.getAbsolutePath())
                tileList << [
                    tile_id: tileId,
                    file_name: tileFile.getName(),
                    file_path: tileFile.getAbsolutePath(),
                    x: x,
                    y: y,
                    width: w,
                    height: h,
                    downsample: downsample
                ]
                tileCount++
            } catch (Exception e) {
                println "[WARN] Failed tile (${x}, ${y}): ${e.getMessage()}"
            }
        }
    }

    // Write image-level metadata
    File metaFile = new File(imgOutputDir, "tiles_metadata.json")
    metaFile.text = new JsonBuilder([
        image_name: imageName,
        pixel_size_um: currentMpp,
        target_pixel_size_um: targetMpp,
        downsample: downsample,
        total_tiles: tileCount,
        tiles: tileList
    ]).toPrettyString()

    masterManifest[imageName] = [
        total_tiles: tileCount,
        metadata_file: metaFile.getAbsolutePath(),
        output_directory: imgOutputDir.getAbsolutePath()
    ]
    println "[SUCCESS] Finished ${imageName}: ${tileCount} tiles exported."
}

// Write master batch manifest
File manifestFile = new File(outputDirBase, "master_batch_manifest.json")
manifestFile.text = new JsonBuilder(masterManifest).toPrettyString()
println "\n=========================================================="
println "BATCH EXPORT COMPLETE: Master Manifest -> ${manifestFile.getAbsolutePath()}"
println "=========================================================="

