import numpy as np
from PIL import Image


def part_a(NxN_Array, KxK_Kernel):
    returningMatrixSize = (NxN_Array.shape[0] - KxK_Kernel.shape[0]) + 1
    returningMatrix = np.empty([returningMatrixSize, returningMatrixSize])
    for iterationOneFromMatrixSize in range(returningMatrixSize):
        for iterationTwoFromMatrixSize in range(returningMatrixSize):
            summation = 0
            for iterationOneFromKernel in range(KxK_Kernel.shape[0]):
                for iterationTwoFromKernel in range(KxK_Kernel.shape[0]):
                    summation = summation + NxN_Array[iterationOneFromKernel + iterationOneFromMatrixSize][
                        iterationTwoFromKernel + iterationTwoFromMatrixSize] * KxK_Kernel[iterationOneFromKernel][
                                    iterationTwoFromKernel]
            returningMatrix[iterationOneFromMatrixSize][iterationTwoFromMatrixSize] = summation
    return returningMatrix


def part_b():
    # Code given in question:
    im = Image.open('pound.png')
    rgb = np.array(im.convert('RGB'))
    r = rgb[:, :, 0]  # array of R pixels
    img_array = np.uint8(r)

    # Using only kernel #1
    firstKernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    imageOneUsingFirstKernel = part_a(img_array, firstKernel)
    Image.fromarray(imageOneUsingFirstKernel).show()

    # Using only kernel #2
    secondKernel = np.array([[0, -1, 0], [-1, 8, -1], [0, -1, 0]])
    imageTwoUsingSecondKernel = part_a(img_array, secondKernel)
    Image.fromarray(imageTwoUsingSecondKernel).show()

    # Using both kernel #1 and kernel #2
    firstAndSecondKernel = part_a(img_array, firstKernel)
    firstAndSecondKernel = part_a(firstAndSecondKernel, secondKernel)
    Image.fromarray(firstAndSecondKernel).show()


if __name__ == '__main__':
    # Uncomment the part you wish to run below
    # part_a()
    part_b()
