

### **Day 13 - Movie Recommendation System (Hybrid Model)**  
This is part of my **#100DaysOfAI** challenge.  
On **Day 13**, I built a **hybrid movie recommendation system** combining both **Collaborative Filtering** and **Content-Based Filtering**. This system recommends movies based on user preferences and movie metadata like tags.

---

### **Goal**  
Build a recommender system that suggests movies to users by learning from:
- **User ratings** using the SVD algorithm (collaborative filtering).
- **Movie tags** using TF-IDF and cosine similarity (content-based filtering).

---

### **Technologies Used**

| Tool        | Purpose                                                     |
|-------------|-------------------------------------------------------------|
| Python      | Main programming language                                   |
| Pandas      | Data manipulation for ratings and tags                      |
| scikit-learn | TF-IDF vectorization and cosine similarity                 |
| Surprise    | Collaborative filtering using SVD                           |
| NumPy       | Data handling and matrix operations                         |
| VS Code | Code development and testing                          |

---

### **How It Works**

1. **Data Loading**
   - Loaded `ratings.csv` and `tags.csv`.
   - Limited to 1000 rows for faster processing.

2. **Preprocessing**
   - Cleaned missing tags.
   - Grouped tags by movie and created dummy movie titles.

3. **Collaborative Filtering (SVD)**
   - Used the `Surprise` library to train an SVD model.
   - Predicted user ratings for unseen movies.
   - Generated top-N movie recommendations per user.

4. **Content-Based Filtering (TF-IDF)**
   - Applied TF-IDF to combined movie tags.
   - Calculated cosine similarity between movies.
   - Recommended similar movies based on tag similarity.

5. **Hybrid Output**
   - Showed SVD-based top 5 recommendations for a specific user.
   - Showed tag-based similar movies for a specific movie (ID = 1).

---

### **Highlights**

- Built a working hybrid recommendation engine.
- Learned to combine **explicit ratings** with **metadata-based similarity**.
- Used `Surprise` for matrix factorization (SVD).
- Practiced **TF-IDF** vectorization and **cosine similarity**.
- Connected collaborative and content-based results meaningfully.

---

### **What I Learned**

- How to build collaborative filtering using SVD and Surprise.
- How to clean and process tag data for TF-IDF modeling.
- How to combine both recommendation strategies in a single pipeline.
- Importance of merging datasets for unified recommendations.
- Real-world relevance of hybrid models used in platforms like Netflix or Amazon.

---

