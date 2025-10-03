import numpy as np
import cv2

# Function to check if the wire passes through a border using sorted strip statistics
def border_has_wire(strip, threshold=40, n=10):
    # Sort the strip values
    sorted_strip = np.sort(strip.flatten())
    # Take the average of the n smallest and n largest values
    avg_smallest = np.mean(sorted_strip[:n])
    avg_largest = np.mean(sorted_strip[-n:])
    diff = abs(avg_largest - avg_smallest)
    print(f"Avg of {n} smallest: {avg_smallest}")
    print(f"Avg of {n} largest: {avg_largest}")
    print(f"Difference: {diff}")
    return diff > threshold


def check_wire_connectivity(snapshot, height, width):
    # Convert to grayscale for easier analysis
    gray = cv2.cvtColor(snapshot, cv2.COLOR_BGR2GRAY)

    # Select a few pixels to calculate differences, number based on image size
    num_pixels = min(10, height//8)

    # Get border strips
    top_strip = gray[0, :]
    bottom_strip = gray[-1, :]
    left_strip = gray[:, 0]
    right_strip = gray[:, -1]

    # Check each border
    borders = {
        'top': border_has_wire(top_strip, n=num_pixels),
        'bottom': border_has_wire(bottom_strip, n=num_pixels),
        'left': border_has_wire(left_strip, n=num_pixels),
        'right': border_has_wire(right_strip, n=num_pixels)
    }

    borders_c_indexed = []
    if borders['left']:
        borders_c_indexed.append(-1)
    if borders['right']:
        borders_c_indexed.append(1)

    borders_r_indexed = []
    if borders['top']:
        borders_r_indexed.append(-1)
    if borders['bottom']:
        borders_r_indexed.append(1)

    print('Wire passes through these borders:')
    for border, present in borders.items():
        if present:
            print(f'- {border}')
    if not any(borders.values()):
        print('No border detected with wire passing through.')

    return (borders_r_indexed, borders_c_indexed)