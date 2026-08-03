from sklearn.neighbors import NearestNeighbors

def get_k_distances(X, k=5):
    """ฟังก์ชันช่วยสำหรับวัดระยะห่าง k-nearest neighbors เพื่อวิเคราะห์ความหนาแน่นของข้อมูล"""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    distances, indices = nbrs.kneighbors(X)
    return distances