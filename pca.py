import pandas as pd
from pandas import read_csv
from sklearn.decomposition import PCA

def readfile(filename):
        dataframe = read_csv(filename, header=None, delim_whitespace=True)
        return dataframe.values

def pca_calculate(vectors):
        vectors = vectors.astype('float32')
        pca = PCA(n_components=100)
        principalComponents = pca.fit_transform(vectors)
        principalDf = pd.DataFrame(data = principalComponents)
        print(pca.explained_variance_ratio_.round(2))
        # Principal Components Weights (Eigenvectors)
        df_pca_loadings = pd.DataFrame(pca.components_)
        print(df_pca_loadings.head())
        return principalDf


def main():
        pca_vec = pca_calculate(readfile('vectors'))
        f_pca = 'PCA_output.txt'
        pca_vec.to_csv(f_pca, header=None, index=None, sep=' ', mode='w')

if __name__=="__main__":
        main()

