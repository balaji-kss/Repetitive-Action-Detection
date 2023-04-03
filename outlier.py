import numpy as np

def remove_outliers(arr):
    q1 = np.quantile(arr, 0.25)
    q3 = np.quantile(arr, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5*iqr
    upper_bound = q3 + 1.5*iqr

    return arr[(arr >= lower_bound) & (arr <= upper_bound)]

# Example usage
data = np.array( [ 611,  585,  599,  614,  667,  612,  634,  651,  646,  658,  645,  613,  644, 76,
  592,  586,  635,  595,  631,  577,  585,  602,  577,  583,  590,  576,  625,  630, 598, 596, 1230, 623,  647,  580,  622])
cleaned_data = remove_outliers(data)
print(cleaned_data)
