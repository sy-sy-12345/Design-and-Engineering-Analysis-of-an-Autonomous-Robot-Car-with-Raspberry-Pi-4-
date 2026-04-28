import cv2
import numpy as np
from picamera2 import Picamera2

"""
Camera
   |
ORB symbol detection
   |
If no ORB match
   |
Contour shape detection
   |
Arrow / Pacman / Star / etc classification
"""

# -----------------------------
# ORB + Matcher Setup
# -----------------------------
orb = cv2.ORB_create(nfeatures=1000) # ORB will detect up to 800 keypoints (important points) in the image
bf = cv2.BFMatcher(cv2.NORM_HAMMING) # finds important points (keypoints) in the image & creates a descriptor (a fingerprint) for each keypoint.

# -----------------------------
# Load Reference Symbols
# -----------------------------
symbol_files = {
    "FingerPrint": "fingerprint.png", #print fingerprint
    "Button": "pressbutton.png", #stop
    "QRcode": "qr.png", #print QR
    "Warning": "warning.png", #stop
    "Recycle": "recycle.png" #turn 360
}

symbols = {}
frame_count = 0
num_corners = 0
approx = None

for name, path in symbol_files.items():
    img = cv2.imread(path, 0)
    if img is None:
        print(f"Error loading {path}")
        continue

    kp, des = orb.detectAndCompute(img, None)

    symbols[name] = {
        "image": img,
        "kp": kp,
        "des": des,
        "shape": img.shape
    }

print("Loaded symbols:", list(symbols.keys()))

# -----------------------------
# Setup Pi Camera
# -----------------------------
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (320, 240)})
picam2.configure(config)
picam2.start()

print("Press 'q' to quit")

def arrowType(approx):
    try:
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
    except cv2.error:
        defects = None  # fallback: just skip convexity defects
            
    M = cv2.moments(approx)

    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
    arrow_tip = None
    max_dist = 0

    for point in approx: #Loops through each point in the contour
        px, py = point[0] #Extracts x and y coordinates

        dist = np.sqrt((px-cx)**2 + (py-cy)**2) #Calculates distance from center

        if dist > max_dist:
            max_dist = dist
            arrow_tip = (px, py)
            
    if arrow_tip is None:
        return "Arrow"

    dx = arrow_tip[0] - cx
    dy = arrow_tip[1] - cy
    
    if abs(dx) > abs(dy):

        if dx < 0:
            direction = "Turn Right"
        else:
            direction = "Turn Left"

    else:

        if dy < 0:
            direction = "Reverse"
        else:
            direction = "Forward"
    
    return direction

def QuadrilateralType(approx):
    pts = approx.reshape(4,2) #array to store coordinates of 4 points
    
    # Compute all distances between corners
    def dist(p1,p2):
        return np.linalg.norm(p1-p2)
    
    dists = [
        (0,1,dist(pts[0], pts[1])),
        (0,2,dist(pts[0], pts[2])),
        (0,3,dist(pts[0], pts[3])),
        (1,2,dist(pts[1], pts[2])),
        (1,3,dist(pts[1], pts[3])),
        (2,3,dist(pts[2], pts[3]))
    ]
    
    # Sort distances descending
    dists.sort(key=lambda x: x[2], reverse=True)
    
    # The two largest distances are diagonals
    d1_idx = dists[0][:2]
    d2_idx = dists[1][:2]
    
    diag1 = pts[d1_idx[1]] - pts[d1_idx[0]]
    diag2 = pts[d2_idx[1]] - pts[d2_idx[0]]
    
    # Check if diagonals are roughly perpendicular
    dot = np.dot(diag1, diag2) #Computes dot product If dot ≈ 0 → diagonals are perpendicular
    
    # Sides lengths
    sides = [dist(pts[i], pts[(i+1)%4]) for i in range(4)] #identify the shape type.
    
    # Diamond: sides roughly equal + diagonals roughly perpendicular
    if (max(sides)-min(sides))/max(sides) < 0.15 and abs(dot) < 0.2*np.linalg.norm(diag1)*np.linalg.norm(diag2): #have almost same length for 4 sides
        return "Diamond"
    
    # Trapezium: exactly one pair of opposite sides roughly equal
    parallel_pair = ((abs(sides[0]-sides[2])/max(sides[0],sides[2]) < 0.15) and (abs(sides[1]-sides[3])/max(sides[1],sides[3]) > 0.15)) or \
                    ((abs(sides[1]-sides[3])/max(sides[1],sides[3]) < 0.15) and (abs(sides[0]-sides[2])/max(sides[0],sides[2]) > 0.15))
    if parallel_pair:
        return "Trapezium"
    


# -----------------------------
# Camera Loop
# -----------------------------
while True:
    #Gets image from camera
    #Converts to grayscale (simpler + faster processing)
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #for ORB to look for keypoints & descriptors 
    kp2, des2 = orb.detectAndCompute(gray, None) ##kp and desc from camera frame
    best_name = None

    if des2 is not None: # Only try matching if descriptors actually exist
        
        symbol = False
        best_good = []
        best_kp1 = None
        best_shape = None
        
        # Compare with each symbol (Loop through known symbols)
        for name, data in symbols.items(): ##returrn key, value
            
            #print(f"Symbol: {name} | Keypoints found: {len(kp)}")
            matches = bf.knnMatch(data["des"], des2, k=2) #finds 2 closest matches for each descriptor
            ##matches=[[m1,n1],[m2,n2],..] --array
            ##each item = [m,n]

            good = []
            for pair in matches:
                if len(pair) < 2:  # skip if less than 2 matches
                    continue
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
      
            print(f"Checking {name}: Ref Points: {len(data['kp'])} | Good Matches: {len(good)}") ##see ref got how many vs good ones how many

            if len(good) > len(best_good):
                best_name = name ##name is for(int i...) the 'i'
                best_good = good
                best_kp1 = data["kp"]
                best_shape = data["shape"]

        # If enough matches found
        if len(best_good) > 12: # change ratio for better performance (change ratio test-> "if m.distance < 0.8 * n.distance:")
            # REF IMG
            src_pts = np.float32( ##np=arr like lists, but faster, can do math operations | float32= data type in np | np.float32 extracts point locations from the ref img
                [best_kp1[m.queryIdx].pt for m in best_good] ##queryIdx = Index of the keypoint in the ref img | m.queryIdx = matched keypoint
            ).reshape(-1, 1, 2) ##changes arr shape w/o change data || print(a.reshape(2,3)) = [[,,] , [,,]] --2rows 3cols || print(a.shape) = (6,) --1D arr with 6 elements 
                ##.reshape(a,b,c) - -1=decide dimensions size yourself , b=rows , c=cols
            
            # LIVE CAM IMG
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in best_good] ##m.trainidx = FOR LIVE CAM IMG
            ).reshape(-1, 1, 2)

            M, mask = cv2.findHomography( ##findHomography = rotation,scaling,translation,perspective tilt
                src_pts, dst_pts, cv2.RANSAC, 5.0 ##ref img,live cam img,removes bad matches (outliers) automatically,allow up to 5 pixels of error when fitting the transformation
            )

            # M = 3x3 matrix

            ####print(f"shape detected: {best_name}!")
            if best_name is not None:
                symbol = True
                
            if M is not None: # found the transformation matrix

                h, w = best_shape
                
                #get symbol corners
                pts = np.float32(
                    [[0,0],[0,h],[w,h],[w,0]] ##creates 4 corner points of the reference image:
                ).reshape(-1,1,2)

                dst = cv2.perspectiveTransform(pts, M)
                frame = cv2.polylines( frame, [np.int32(dst)], True, (0,255,0), 3)

                cv2.putText( ##writes text "Detected: {best_name}"
                    frame,
                    f"Detected: {best_name}",
                    (30,40), ##write on this coordinate of the window display
                    cv2.FONT_HERSHEY_SIMPLEX, ##font type
                    1, ##font scale
                    (0,255,0), ##GREEN
                    2 ##thickness
                )
                                    
                            
    #compare color and contour
    #Converts image → binary → edges

    #If pixel value > 150 → make it 255 (white)
    #If pixel value ≤ 150 → make it 0 (black)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY) #converts a grayscale image into a black-and-white (binary) image

    # 1. Prepare the image (Blur and Edge detection)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(thresh, 30, 100) #.Canny for edge detection

    # 2. Find the outlines
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:         
        shapename = "Unknown"
        area = cv2.contourArea(cnt)
        
        # Ignore tiny noise
        if cv2.contourArea(cnt) < 800:
            continue
        
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)

        #Solidity → how filled the shape is
        #Extent → how much of bounding box is filled
        #Circularity → how circle-like it is
        solidity = area / hull_area if hull_area != 0 else 0

        x,y,w,h = cv2.boundingRect(cnt)
        rect_area = w*h
        extent = area / rect_area if rect_area != 0 else 0

        perimeter = cv2.arcLength(cnt, True)
        circularity = 4*np.pi*area/(perimeter*perimeter) if perimeter!=0 else 0

        # 3. Simplify the shape to find corners
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        num_corners = len(approx)
        #print(f"num_corner: {num_corners}")  
        # Identify by corner count
        if 10 <= num_corners <= 11:
            shapename = "Star"
        elif num_corners == 12:
            shapename = "Plus"
        elif num_corners == 8 and extent > 0.70:
            shapename = "Octagon"
        elif 6 <= num_corners <= 12 and 0.70 < solidity < 0.92 and extent > 0.55:
            shapename = "Pacman"
        elif 8 <= num_corners <= 9:
            shapename = arrowType(approx)
        elif 6 <= num_corners <= 9 and solidity >= 0.95 and circularity < 0.88:
            shapename = "Segmented Circle"
        elif num_corners == 4:
            shapename = QuadrilateralType(approx)
                      
        arr = None         
        ####print(f"shape detected: {shapename}!")
        if symbol:
            print(f"SYMBOL DETECTED: {best_name}")
        else:
            cv2.drawContours(frame, [approx], -1, (0, 255, 0), 3)
            cv2.putText(frame, shapename, (approx[0][0][0], approx[0][0][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            print(f"SHAPE DETECTED: {shapename}")
            
    
    cv2.imshow("Symbol Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
picam2.stop()