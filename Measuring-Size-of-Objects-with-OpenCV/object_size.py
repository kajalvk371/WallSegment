import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_walls(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None
    
    # Create a copy for visualization
    output = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to reduce noise while preserving edges
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # ENHANCEMENT 1: Use adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Canny edge detection with optimized parameters
    edges = cv2.Canny(filtered, 30, 130)  # Lowered threshold to catch more edges
    
    # ENHANCEMENT 2: Combine thresholding and edge detection for better results
    combined_mask = cv2.bitwise_or(thresh, edges)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((5, 5), np.uint8)  # Increased kernel size
    dilated_edges = cv2.dilate(combined_mask, kernel, iterations=3)  # More iterations
    
    # ENHANCEMENT 3: Apply additional morphological operations
    # Close small holes
    closed = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, 
                             kernel=np.ones((15, 15), np.uint8))
    
    # Find lines using Hough Line Transform with more sensitive parameters
    lines = cv2.HoughLinesP(closed, 1, np.pi/180, threshold=40,  # Lower threshold
                           minLineLength=40, maxLineGap=30)      # More forgiving parameters
    
    # Create a blank mask for wall candidates
    wall_mask = np.zeros_like(gray)
    
    # Draw detected lines on mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 7)  # Thicker lines
    
    # Apply morphological closing to connect nearby lines
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, 
                                kernel=np.ones((21, 21), np.uint8))  # Larger kernel
    
    # ENHANCEMENT 4: Apply multi-scale contour detection
    multi_scale_mask = np.zeros_like(gray)
    
    # Process at multiple scales for better detection
    for scale in [0.5, 0.75, 1.0]:
        if scale != 1.0:
            scaled_img = cv2.resize(closed, None, fx=scale, fy=scale)
            scaled_contours, _ = cv2.findContours(scaled_img, cv2.RETR_EXTERNAL, 
                                                 cv2.CHAIN_APPROX_SIMPLE)
            
            # Scale contours back to original size
            for contour in scaled_contours:
                contour = (contour / scale).astype(np.int32)
                cv2.drawContours(multi_scale_mask, [contour], -1, 255, thickness=cv2.FILLED)
        else:
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, 
                                          cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                cv2.drawContours(multi_scale_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Combine multi-scale mask with wall mask
    combined_wall_mask = cv2.bitwise_or(wall_mask, multi_scale_mask)
    
    # Find contours in the combined wall mask
    contours, _ = cv2.findContours(combined_wall_mask, cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)
    
    # Create refined wall mask
    refined_mask = np.zeros_like(gray)
    
    # ENHANCEMENT 5: Improved wall detection criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate circularity (for walls, this should be low)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate aspect ratio
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Calculate solidity (area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Walls typically have large area, low circularity, and high aspect ratio or high solidity
        if ((area > 800 and                     # Lower area threshold to catch smaller wall segments
             circularity < 0.9 and             # More permissive circularity
             (aspect_ratio > 1.2 or             # Lower aspect ratio requirement
              solidity > 0.7)) or              # Consider highly solid regions
            (area > 5000)):                     # Large areas are likely walls
            cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # ENHANCEMENT 6: Apply region-based segmentation to refine wall areas
    # Use grabcut algorithm for final refinement
    grabcut_mask = np.zeros(image.shape[:2], np.uint8)
    grabcut_mask[:] = cv2.GC_PR_BGD  # Set all to probable background
    
    # Set refined wall areas as probable foreground
    grabcut_mask[refined_mask == 255] = cv2.GC_PR_FGD
    
    # GrabCut parameters
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Run GrabCut
    cv2.grabCut(image, grabcut_mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
    
    # Create final mask where definite and probable foreground are set to white
    final_mask = np.zeros_like(gray)
    final_mask[np.where((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD))] = 255
    
    # ENHANCEMENT 7: Post-processing to remove small artifacts and fill holes
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, 
                                 kernel=np.ones((3, 3), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, 
                                 kernel=np.ones((11, 11), np.uint8))
    
    # Create final output visualization
    colored_mask = np.zeros_like(image)
    colored_mask[final_mask == 255] = [0, 255, 0]  # Green for walls
    
    # Blend with original image
    alpha = 0.6
    beta = 1 - alpha
    segmented_image = cv2.addWeighted(image, alpha, colored_mask, beta, 0)
    
    # Draw contours on the output image
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(segmented_image, contours, -1, (0, 0, 255), 2)
    
    return final_mask, segmented_image

def calculate_wall_area(mask, pixel_to_meter_ratio=0.01):
    """Calculate the wall area in square meters"""
    # Count white pixels (walls)
    wall_pixels = np.sum(mask == 255)
    
    # Convert to real-world area
    wall_area = wall_pixels * (pixel_to_meter_ratio ** 2)
    return wall_area

def estimate_paint_cost(image_path, cost_per_sqm=3, pixel_to_meter_ratio=0.01):
    """Estimate the cost of painting walls"""
    wall_mask, segmented_image = segment_walls(image_path)
    if wall_mask is None:
        return None, None, None, None
    
    # ENHANCEMENT 8: Calculate wall area with additional statistics
    wall_pixels = np.sum(wall_mask == 255)
    wall_area = calculate_wall_area(wall_mask, pixel_to_meter_ratio)
    
    # Calculate coverage statistics
    image_size = wall_mask.shape[0] * wall_mask.shape[1]
    wall_coverage = (wall_pixels / image_size) * 100
    
    # ENHANCEMENT 9: Cost calculation with paint efficiency factor
    # Assuming standard paint covers approximately 10-12 m² per liter
    # and each coat requires 2 applications for proper coverage
    coats = 2  # Standard number of coats
    paint_coverage_per_liter = 11  # m² per liter (average)
    liters_needed = (wall_area / paint_coverage_per_liter) * coats
    
    # Calculate material cost
    paint_cost_per_liter = cost_per_sqm * paint_coverage_per_liter
    material_cost = liters_needed * paint_cost_per_liter
    
    # Calculate labor cost (typically 30-40% of material cost)
    labor_cost = material_cost * 0.35  # 35% of material cost
    
    # Total cost including materials and labor
    total_cost = material_cost + labor_cost
    
    # ENHANCEMENT 10: Return comprehensive cost breakdown
    cost_details = {
        "wall_area_sqm": wall_area,
        "wall_coverage_percent": wall_coverage,
        "paint_liters_needed": liters_needed,
        "material_cost": material_cost,
        "labor_cost": labor_cost,
        "total_cost": total_cost
    }
    
    return total_cost, segmented_image, wall_area, cost_details

def display_results(image_path, pixel_to_meter_ratio=0.01, cost_per_sqm=3):
    """Display segmentation results with detailed information"""
    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Get segmentation results
    total_cost, segmented, area, cost_details = estimate_paint_cost(
        image_path, cost_per_sqm, pixel_to_meter_ratio)
    
    if total_cost is None:
        return
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    # Segmented image
    plt.subplot(2, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title('Segmented Walls')
    plt.axis('off')
    
    # ENHANCEMENT 11: Cost breakdown chart
    plt.subplot(2, 2, 3)
    costs = ['Material', 'Labor']
    amounts = [cost_details['material_cost'], cost_details['labor_cost']]
    bars = plt.bar(costs, amounts, color=['#3498db', '#2ecc71'])
    plt.title('Cost Breakdown')
    plt.ylabel('Cost (Rs)')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'Rs {height:.2f}', ha='center', va='bottom')
    
    # ENHANCEMENT 12: Wall area visualization
    plt.subplot(2, 2, 4)
    plt.text(0.5, 0.5, f"Total Wall Area: {area:.2f} m²\n"
             f"Paint Required: {cost_details['paint_liters_needed']:.2f} liters\n"
             f"Wall Coverage: {cost_details['wall_coverage_percent']:.1f}% of image\n"
             f"Total Cost: Rs {total_cost:.2f}",
             horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=12,
             bbox=dict(boxstyle='round,pad=1', facecolor='#f0f0f0', alpha=1))
    plt.axis('off')
    plt.title('Paint Estimation Details')
    
    plt.tight_layout()
    
    # Save results
    results_dir = "wall_segmentation_results"
    os.makedirs(results_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    plt.savefig(f"{results_dir}/{base_name}_analysis.png", dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print detailed report
    print("\n" + "="*50)
    print(f"WALL PAINT COST ESTIMATION REPORT FOR: {os.path.basename(image_path)}")
    print("="*50)
    print(f"Total Wall Area: {area:.2f} square meters")
    print(f"Paint Required: {cost_details['paint_liters_needed']:.2f} liters")
    print(f"Material Cost: Rs {cost_details['material_cost']:.2f}")
    print(f"Labor Cost: Rs {cost_details['labor_cost']:.2f}")
    print(f"TOTAL ESTIMATED COST: Rs {total_cost:.2f}")
    print("="*50)
    
    return total_cost, area, cost_details

def batch_process_images(image_folder, pixel_to_meter_ratio=0.01, cost_per_sqm=3):
    """Process multiple images in a folder and generate a summary report"""
    if not os.path.exists(image_folder):
        print(f"Error: Folder {image_folder} does not exist")
        return
        
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"No image files found in {image_folder}")
        return
        
    results = []
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        print(f"Processing {image_file}...")
        
        total_cost, segmented, area, cost_details = estimate_paint_cost(
            image_path, cost_per_sqm, pixel_to_meter_ratio)
            
        if total_cost is not None:
            results.append({
                "image": image_file,
                "wall_area": area,
                "total_cost": total_cost,
                "details": cost_details
            })
            
            # Save segmented image
            results_dir = "wall_segmentation_results"
            os.makedirs(results_dir, exist_ok=True)
            base_name = os.path.basename(image_path).split('.')[0]
            cv2.imwrite(f"{results_dir}/{base_name}_segmented.jpg", segmented)
    
    # Generate summary report
    if results:
        print("\n" + "="*60)
        print("WALL PAINT ESTIMATION SUMMARY REPORT")
        print("="*60)
        print(f"{'Image':<30} {'Wall Area (m²)':<15} {'Cost (Rs)':<15}")
        print("-"*60)
        
        total_area = 0
        total_cost = 0
        
        for result in results:
            print(f"{result['image']:<30} {result['wall_area']:<15.2f} {result['total_cost']:<15.2f}")
            total_area += result['wall_area']
            total_cost += result['total_cost']
            
        print("-"*60)
        print(f"{'TOTAL':<30} {total_area:<15.2f} {total_cost:<15.2f}")
        print("="*60)
        
        # Create and save summary visualization
        plt.figure(figsize=(10, 8))
        
        # Bar chart of costs per image
        plt.subplot(2, 1, 1)
        image_names = [r['image'] for r in results]
        costs = [r['total_cost'] for r in results]
        plt.bar(image_names, costs, color='#3498db')
        plt.title('Cost Estimation by Room')
        plt.ylabel('Cost (Rs)')
        plt.xticks(rotation=45, ha='right')
        
        # Pie chart of area distribution
        plt.subplot(2, 1, 2)
        areas = [r['wall_area'] for r in results]
        plt.pie(areas, labels=image_names, autopct='%1.1f%%', startangle=90)
        plt.title('Wall Area Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/summary_report.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Process single image
    image_path = "images/image2.jpg"  # Replace with your image path
    display_results(image_path, pixel_to_meter_ratio=0.01, cost_per_sqm=3)
    
    # Uncomment to process all images in a folder
    # batch_process_images("images", pixel_to_meter_ratio=0.01, cost_per_sqm=3)