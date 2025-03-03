import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime  # Import datetime properly

def segment_walls(image_path):
    """Segment walls from room images using improved computer vision techniques"""
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
    
    # Use adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Canny edge detection with optimized parameters
    edges = cv2.Canny(filtered, 30, 130)
    
    # Combine thresholding and edge detection for better results
    combined_mask = cv2.bitwise_or(thresh, edges)
    
    # Dilate edges to connect broken lines
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(combined_mask, kernel, iterations=3)
    
    # Close small holes
    closed = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, 
                             kernel=np.ones((15, 15), np.uint8))
    
    # Find lines using Hough Line Transform
    lines = cv2.HoughLinesP(closed, 1, np.pi/180, threshold=40,
                           minLineLength=40, maxLineGap=30)
    
    # Create a blank mask for wall candidates
    wall_mask = np.zeros_like(gray)
    
    # Draw detected lines on mask
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(wall_mask, (x1, y1), (x2, y2), 255, 7)
    
    # Apply morphological closing to connect nearby lines
    wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, 
                                kernel=np.ones((21, 21), np.uint8))
    
    # Apply multi-scale contour detection
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create refined wall mask
    refined_mask = np.zeros_like(gray)
    
    # Apply improved wall detection criteria
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Calculate features to identify walls
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Filter for wall-like structures
        if ((area > 800 and circularity < 0.9 and (aspect_ratio > 1.2 or solidity > 0.7)) or
            (area > 5000)):
            cv2.drawContours(refined_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    # Post-process the mask
    final_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, 
                                 kernel=np.ones((3, 3), np.uint8))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, 
                                 kernel=np.ones((11, 11), np.uint8))
    
    # Create visualization
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
    wall_pixels = np.sum(mask == 255)
    wall_area = wall_pixels * (pixel_to_meter_ratio ** 2)
    return wall_area

def calculate_wall_height(image_shape, room_height=2.8):
    """Estimate wall height based on image shape and typical room height"""
    # If not provided, estimate the wall height based on typical room dimensions
    return room_height  # default room height in meters

def estimate_paint_requirements(wall_area, height=2.8):
    """Calculate paint requirements based on wall area"""
    # Consider the typical wall height when estimating total wall surface
    # Assume the image captures most of the wall width but not full height
    
    # Calculate total paintable area (including estimates for full height)
    paintable_area = wall_area
    
    # Return the estimated area
    return paintable_area

def estimate_paint_cost(image_path, room_height=2.8, pixel_to_meter_ratio=0.01):
    """Estimate the cost of painting walls with realistic pricing"""
    # Current market rates (as of 2025)
    PAINT_PRICES = {
        "economy": {"price_per_liter": 350, "coverage_sqm_per_liter": 10, "num_coats": 2},
        "standard": {"price_per_liter": 650, "coverage_sqm_per_liter": 12, "num_coats": 2},
        "premium": {"price_per_liter": 1200, "coverage_sqm_per_liter": 14, "num_coats": 2},
    }
    
    # Labor costs (per square meter in INR)
    LABOR_RATES = {
        "economy": 40,  # Basic painting
        "standard": 60,  # Standard painting with basic preparation
        "premium": 100,  # Premium service with thorough preparation
    }
    
    # Additional materials (primer, putty, etc.) as percentage of paint cost
    ADDITIONAL_MATERIALS = {
        "economy": 0.15,  # 15% of paint cost
        "standard": 0.20,  # 20% of paint cost
        "premium": 0.25,  # 25% of paint cost
    }
    
    # Segment walls from the image
    wall_mask, segmented_image = segment_walls(image_path)
    if wall_mask is None:
        return None, None, None
    
    # Calculate wall area and adjust for the room height
    estimated_area = calculate_wall_area(wall_mask, pixel_to_meter_ratio)
    wall_height = calculate_wall_height(wall_mask.shape, room_height)
    
    # Estimate actual paintable area
    paintable_area = estimate_paint_requirements(estimated_area, wall_height)
    
    # Calculate costs for different quality options
    cost_options = {}
    
    for quality in ["economy", "standard", "premium"]:
        paint_spec = PAINT_PRICES[quality]
        labor_rate = LABOR_RATES[quality]
        materials_factor = ADDITIONAL_MATERIALS[quality]
        
        # Calculate paint quantity needed
        liters_needed = (paintable_area / paint_spec["coverage_sqm_per_liter"]) * paint_spec["num_coats"]
        
        # Calculate costs
        paint_cost = liters_needed * paint_spec["price_per_liter"]
        additional_materials_cost = paint_cost * materials_factor
        labor_cost = paintable_area * labor_rate
        
        # Total cost
        total_cost = paint_cost + additional_materials_cost + labor_cost
        
        # Save details
        cost_options[quality] = {
            "paint_liters": liters_needed,
            "paint_cost": paint_cost,
            "materials_cost": additional_materials_cost,
            "labor_cost": labor_cost,
            "total_cost": total_cost
        }
    
    # Create detailed cost breakdown
    cost_details = {
        "wall_area_sqm": paintable_area,
        "options": cost_options
    }
    
    return cost_details, segmented_image, paintable_area

def display_results(image_path, room_height=2.8, pixel_to_meter_ratio=0.01):
    """Display segmentation results with detailed cost breakdown"""
    # Load original image
    original = cv2.imread(image_path)
    if original is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Get segmentation results
    cost_details, segmented, area = estimate_paint_cost(
        image_path, room_height, pixel_to_meter_ratio)
    
    if cost_details is None:
        return
    
    # Convert BGR to RGB for matplotlib
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    segmented_rgb = cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 12))
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_rgb)
    plt.title('Original Room Image')
    plt.axis('off')
    
    # Segmented image
    plt.subplot(2, 2, 2)
    plt.imshow(segmented_rgb)
    plt.title('Detected Wall Areas')
    plt.axis('off')
    
    # Cost breakdown chart
    plt.subplot(2, 2, 3)
    
    # Extract costs for different quality options
    qualities = list(cost_details["options"].keys())
    paint_costs = [cost_details["options"][q]["paint_cost"] for q in qualities]
    material_costs = [cost_details["options"][q]["materials_cost"] for q in qualities]
    labor_costs = [cost_details["options"][q]["labor_cost"] for q in qualities]
    
    # Create stacked bar chart
    bar_width = 0.6
    plt.bar(qualities, paint_costs, bar_width, label='Paint', color='#3498db')
    plt.bar(qualities, material_costs, bar_width, bottom=paint_costs, label='Materials', color='#2ecc71')
    
    # Calculate the bottom position for labor costs
    bottom_labor = [p + m for p, m in zip(paint_costs, material_costs)]
    plt.bar(qualities, labor_costs, bar_width, bottom=bottom_labor, label='Labor', color='#e74c3c')
    
    plt.title('Cost Breakdown by Quality')
    plt.ylabel('Cost (rs)')
    plt.legend()
    
    # Add value labels for total costs
    for i, quality in enumerate(qualities):
        total = cost_details["options"][quality]["total_cost"]
        plt.text(i, total + 500, f'rs {total:.2f}', ha='center', va='bottom')
    
    # Detailed text information
    plt.subplot(2, 2, 4)
    
    # Create detailed text box with all relevant information
    info_text = (
        f"WALL PAINTING COST ESTIMATION\n"
        f"----------------------------------\n"
        f"Total Wall Area: {area:.2f} m²\n\n"
        f"ECONOMY OPTION:\n"
        f" - Paint Required: {cost_details['options']['economy']['paint_liters']:.2f} liters\n"
        f" - Paint Cost: rs{cost_details['options']['economy']['paint_cost']:.2f}\n"
        f" - Materials: rs{cost_details['options']['economy']['materials_cost']:.2f}\n"
        f" - Labor: rs{cost_details['options']['economy']['labor_cost']:.2f}\n"
        f" - Total: rs{cost_details['options']['economy']['total_cost']:.2f}\n\n"
        f"STANDARD OPTION:\n"
        f" - Paint Required: {cost_details['options']['standard']['paint_liters']:.2f} liters\n"
        f" - Paint Cost: rs{cost_details['options']['standard']['paint_cost']:.2f}\n"
        f" - Materials: rs{cost_details['options']['standard']['materials_cost']:.2f}\n"
        f" - Labor: rs{cost_details['options']['standard']['labor_cost']:.2f}\n"
        f" - Total: rs{cost_details['options']['standard']['total_cost']:.2f}\n\n"
        f"PREMIUM OPTION:\n"
        f" - Paint Required: {cost_details['options']['premium']['paint_liters']:.2f} liters\n"
        f" - Paint Cost: rs{cost_details['options']['premium']['paint_cost']:.2f}\n"
        f" - Materials: rs{cost_details['options']['premium']['materials_cost']:.2f}\n"
        f" - Labor: rs{cost_details['options']['premium']['labor_cost']:.2f}\n"
        f" - Total: rs{cost_details['options']['premium']['total_cost']:.2f}"
    )
    
    plt.text(0, 0.5, info_text, fontsize=9, 
             bbox=dict(boxstyle='round,pad=1', facecolor='#f8f9fa', alpha=1),
             va='center')
    plt.axis('off')
    plt.title('Detailed Cost Breakdown')
    
    plt.tight_layout()
    
    # Save results
    results_dir = "wall_analysis_results"
    os.makedirs(results_dir, exist_ok=True)
    base_name = os.path.basename(image_path).split('.')[0]
    plt.savefig(f"{results_dir}/{base_name}_analysis.png", dpi=300, bbox_inches='tight')
    
    # Generate a detailed report as text file
    with open(f"{results_dir}/{base_name}_report.txt", "w") as f:
        f.write(f"WALL PAINTING COST ESTIMATION REPORT\n")
        f.write(f"===================================\n\n")
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d')}\n\n")  # Fixed datetime usage
        f.write(f"WALL MEASUREMENTS:\n")
        f.write(f"Total Detected Wall Area: {area:.2f} m²\n\n")
        f.write(f"COST BREAKDOWN:\n\n")
        
        for quality in ["economy", "standard", "premium"]:
            option = cost_details["options"][quality]
            f.write(f"{quality.upper()} QUALITY OPTION:\n")
            f.write(f"- Paint Required: {option['paint_liters']:.2f} liters\n")
            f.write(f"- Paint Cost: rs{option['paint_cost']:.2f}\n")
            f.write(f"- Additional Materials: rs{option['materials_cost']:.2f}\n")
            f.write(f"- Labor Cost: rs{option['labor_cost']:.2f}\n")
            f.write(f"- TOTAL COST: rs{option['total_cost']:.2f}\n\n")
    
    plt.show()
    
    # Print summary to console
    print("\n" + "="*50)
    print(f"WALL PAINTING COST ESTIMATION FOR: {os.path.basename(image_path)}")
    print("="*50)
    print(f"Total Wall Area: {area:.2f} square meters\n")
    
    print("COST OPTIONS:")
    for quality in ["economy", "standard", "premium"]:
        option = cost_details["options"][quality]
        print(f"{quality.upper()}: rs{option['total_cost']:.2f} (Paint: rs{option['paint_cost']:.2f}, "
              f"Materials: rs{option['materials_cost']:.2f}, Labor: rs{option['labor_cost']:.2f})")
    
    print("="*50)
    
    return cost_details, area

def batch_process_images(image_folder, room_height=2.8, pixel_to_meter_ratio=0.01):
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
        
        cost_details, segmented, area = estimate_paint_cost(
            image_path, room_height, pixel_to_meter_ratio)
            
        if cost_details is not None:
            results.append({
                "image": image_file,
                "wall_area": area,
                "cost_details": cost_details
            })
            
            # Save segmented image
            results_dir = "wall_analysis_results"
            os.makedirs(results_dir, exist_ok=True)
            base_name = image_file.split('.')[0]
            cv2.imwrite(f"{results_dir}/{base_name}_segmented.jpg", segmented)
    
    # Generate summary report
    if results:
        # Create directory for results
        results_dir = "wall_analysis_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Create CSV summary
        with open(f"{results_dir}/summary_report.csv", "w") as f:
            f.write("Image,Wall Area (m²),Economy Cost (rs),Standard Cost (rs),Premium Cost (rs)\n")
            
            total_area = 0
            total_costs = {"economy": 0, "standard": 0, "premium": 0}
            
            for result in results:
                economy_cost = result["cost_details"]["options"]["economy"]["total_cost"]
                standard_cost = result["cost_details"]["options"]["standard"]["total_cost"]
                premium_cost = result["cost_details"]["options"]["premium"]["total_cost"]
                
                f.write(f"{result['image']},{result['wall_area']:.2f},{economy_cost:.2f},{standard_cost:.2f},{premium_cost:.2f}\n")
                
                total_area += result['wall_area']
                total_costs["economy"] += economy_cost
                total_costs["standard"] += standard_cost
                total_costs["premium"] += premium_cost
                
            f.write(f"TOTAL,{total_area:.2f},{total_costs['economy']:.2f},{total_costs['standard']:.2f},{total_costs['premium']:.2f}\n")
        
        # Create visualization for summary
        plt.figure(figsize=(15, 10))
        
        # Bar chart of costs per image
        plt.subplot(2, 1, 1)
        image_names = [r['image'] for r in results]
        
        # Get costs for each quality level
        economy_costs = [r["cost_details"]["options"]["economy"]["total_cost"] for r in results]
        standard_costs = [r["cost_details"]["options"]["standard"]["total_cost"] for r in results]
        premium_costs = [r["cost_details"]["options"]["premium"]["total_cost"] for r in results]
        
        # Plot bars
        x = np.arange(len(image_names))
        width = 0.25
        
        plt.bar(x - width, economy_costs, width, label='Economy', color='#3498db')
        plt.bar(x, standard_costs, width, label='Standard', color='#2ecc71')
        plt.bar(x + width, premium_costs, width, label='Premium', color='#e74c3c')
        
        plt.xlabel('Rooms')
        plt.ylabel('Total Cost (rs)')
        plt.title('Painting Cost Comparison by Room')
        plt.xticks(x, image_names, rotation=45, ha='right')
        plt.legend()
        
        # Pie chart of area distribution
        plt.subplot(2, 1, 2)
        areas = [r['wall_area'] for r in results]
        plt.pie(areas, labels=image_names, autopct='%1.1f%%', startangle=90)
        plt.title('Wall Area Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/summary_visualization.png", dpi=300, bbox_inches='tight')
        
        # Print summary to console
        print("\n" + "="*60)
        print("WALL PAINTING COST ESTIMATION SUMMARY REPORT")
        print("="*60)
        print(f"{'Room':<20} {'Area (m²)':<10} {'Economy (rs)':<15} {'Standard (rs)':<15} {'Premium (rs)':<15}")
        print("-"*75)
        
        for result in results:
            economy_cost = result["cost_details"]["options"]["economy"]["total_cost"]
            standard_cost = result["cost_details"]["options"]["standard"]["total_cost"]
            premium_cost = result["cost_details"]["options"]["premium"]["total_cost"]
            
            print(f"{result['image']:<20} {result['wall_area']:<10.2f} {economy_cost:<15.2f} {standard_cost:<15.2f} {premium_cost:<15.2f}")
            
        print("-"*75)
        print(f"{'TOTAL':<20} {total_area:<10.2f} {total_costs['economy']:<15.2f} {total_costs['standard']:<15.2f} {total_costs['premium']:<15.2f}")
        print("="*75)
        
        return results

if __name__ == "__main__":
    # Process single image
    image_path = "images/image.jpg"  # Replace with your image path
    display_results(image_path, room_height=2.8, pixel_to_meter_ratio=0.01)
    
    # Uncomment to process all images in a folder
    # batch_process_images("images", room_height=2.8, pixel_to_meter_ratio=0.01)