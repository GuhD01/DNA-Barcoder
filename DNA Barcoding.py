import tkinter as tk
from tkinter import filedialog, messagebox, Scale, HORIZONTAL
from sklearn.decomposition import PCA
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
from Bio import Phylo

class DNA_Barcoding_App:
    def __init__(self, root):
        self.root = root
        self.root.title("DNA Barcoding Clustering App")
        self.root.geometry("600x850")

        # Title label
        self.title_label = tk.Label(root, text="DNA Barcoding and Clustering", font=("Arial", 16))
        self.title_label.pack(pady=10)

        # File selection
        self.file_button = tk.Button(root, text="Select DNA FASTA File", command=self.load_fasta)
        self.file_button.pack(pady=10)

        # Clustering method label
        self.method_label = tk.Label(root, text="Select Clustering Method:")
        self.method_label.pack(pady=5)

        # Clustering method selection
        self.clustering_method_var = tk.StringVar()
        self.clustering_method_var.set("Threshold Clustering")  # Default value

        self.method_options = ["Threshold Clustering", "K-Means Clustering"]
        self.method_menu = tk.OptionMenu(root, self.clustering_method_var, *self.method_options)
        self.method_menu.pack(pady=5)

        # Number of clusters label
        self.num_clusters_label = tk.Label(root, text="Number of Clusters (K):")
        self.num_clusters_label.pack(pady=5)

        # Number of clusters entry
        self.num_clusters_entry = tk.Entry(root)
        self.num_clusters_entry.insert(0, "3")  # Default value
        self.num_clusters_entry.pack(pady=5)

        # Automatic cluster determination button
        self.auto_cluster_button = tk.Button(root, text="Determine Optimal K", state=tk.DISABLED, command=self.determine_optimal_clusters)
        self.auto_cluster_button.pack(pady=10)

        # Clustering button
        self.cluster_button = tk.Button(root, text="Cluster DNA Sequences", state=tk.DISABLED, command=self.cluster_sequences)
        self.cluster_button.pack(pady=10)

        # Barcode Gap Analysis button
        self.barcode_gap_button = tk.Button(root, text="Perform Barcode Gap Analysis", state=tk.DISABLED, command=self.barcode_gap_analysis)
        self.barcode_gap_button.pack(pady=10)

        # Save results button
        self.save_button = tk.Button(root, text="Save Results", state=tk.DISABLED, command=self.save_results)
        self.save_button.pack(pady=10)

        # Phylogenetic tree button
        self.tree_button = tk.Button(root, text="Construct Phylogenetic Tree", state=tk.DISABLED, command=self.construct_tree)
        self.tree_button.pack(pady=10)

        # Textbox to display clustering summary
        self.cluster_summary_text = tk.Text(root, height=10, width=70)
        self.cluster_summary_text.pack(pady=10)

    def load_fasta(self):
        file_path = filedialog.askopenfilename(filetypes=[("FASTA Files", "*.fasta")])
        if not file_path:
            messagebox.showerror("Error", "No file selected")
            return

        try:
            self.sequences = list(SeqIO.parse(file_path, "fasta"))
            if not self.sequences:
                raise ValueError("No sequences found in the file")
            messagebox.showinfo("Success", f"Loaded {len(self.sequences)} sequences successfully")
            self.cluster_button.config(state=tk.NORMAL)
            self.tree_button.config(state=tk.NORMAL)
            self.auto_cluster_button.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load FASTA file: {e}")

    def determine_optimal_clusters(self):
        try:
            if not hasattr(self, 'sequences') or not self.sequences:
                raise ValueError("No sequences loaded. Please load a valid FASTA file.")

            dna_texts = [str(record.seq) for record in self.sequences]
            if not dna_texts:
                raise ValueError("No valid DNA sequences available for clustering.")

            # Convert DNA sequences to numerical format using CountVectorizer
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
            X = vectorizer.fit_transform(dna_texts).toarray()

            # Determine the optimal number of clusters using the Elbow Method and Silhouette Analysis
            wcss = []
            silhouette_scores = []
            cluster_range = range(2, min(11, len(X)))  # Reasonable range for K
            for k in cluster_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
                score = silhouette_score(X, kmeans.labels_)
                silhouette_scores.append(score)

            # Plot WCSS (Elbow Method)
            plt.figure(figsize=(10, 5))
            plt.plot(cluster_range, wcss, marker='o')
            plt.title('Elbow Method for Optimal K')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
            plt.xticks(cluster_range)
            plt.show()

            # Plot Silhouette Scores
            plt.figure(figsize=(10, 5))
            plt.plot(cluster_range, silhouette_scores, marker='o')
            plt.title('Silhouette Scores for Different K')
            plt.xlabel('Number of clusters (K)')
            plt.ylabel('Average Silhouette Score')
            plt.xticks(cluster_range)
            plt.show()

            # Suggest the optimal K based on maximum silhouette score
            optimal_k = cluster_range[np.argmax(silhouette_scores)]
            messagebox.showinfo("Optimal K Determined", f"The suggested optimal number of clusters is {optimal_k}.")
            # Update the number of clusters entry
            self.num_clusters_entry.delete(0, tk.END)
            self.num_clusters_entry.insert(0, str(optimal_k))

        except Exception as e:
            messagebox.showerror("Error", f"Optimal cluster determination failed: {e}")

    def cluster_sequences(self):
        try:
            # Extract sequences in text format for feature extraction
            if not hasattr(self, 'sequences') or not self.sequences:
                raise ValueError("No sequences loaded. Please load a valid FASTA file.")

            dna_texts = [str(record.seq) for record in self.sequences]
            if not dna_texts:
                raise ValueError("No valid DNA sequences available for clustering.")

            # Convert DNA sequences to numerical format using CountVectorizer
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
            X = vectorizer.fit_transform(dna_texts).toarray()

            # Calculate pairwise distances using Euclidean distance
            distances = np.zeros((len(X), len(X)))
            for i, j in itertools.combinations(range(len(X)), 2):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist

            # Store the distance matrix for later use in tree construction and barcode gap analysis
            self.distance_matrix = distances

            # Get the selected clustering method
            method = self.clustering_method_var.get()

            if method == "Threshold Clustering":
                # Get threshold value from entry
                threshold_percentile = float(self.num_clusters_entry.get())
                threshold = np.percentile(distances, threshold_percentile)  # Convert slider value to a threshold percentile

                # Apply a simple clustering approach based on thresholding distances
                clusters = [-1] * len(X)
                current_cluster = 0

                for i in range(len(X)):
                    if clusters[i] == -1:  # If not assigned to a cluster
                        clusters[i] = current_cluster
                        for j in range(i + 1, len(X)):
                            if distances[i, j] < threshold:
                                clusters[j] = current_cluster
                        current_cluster += 1

                self.num_clusters = len(set(clusters))
                self.cluster_labels = clusters  # Store cluster labels for use in tree construction

                # Visualize the clustering results
                self.visualize_clusters(X, clusters, self.num_clusters, dna_texts)

            elif method == "K-Means Clustering":
                # Get the number of clusters from user input
                k = int(self.num_clusters_entry.get())
                if k <= 0:
                    raise ValueError("Number of clusters must be a positive integer.")

                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X)
                clusters = kmeans.labels_
                self.num_clusters = k
                self.cluster_labels = clusters.tolist()

                # Visualize the clustering results
                self.visualize_clusters(X, clusters, self.num_clusters, dna_texts)

            else:
                raise ValueError("Invalid clustering method selected.")

            # Display cluster summary in the text box
            self.display_cluster_summary(self.cluster_labels, self.num_clusters)

            # Enable Save and Barcode Gap Analysis buttons
            self.save_button.config(state=tk.NORMAL)
            self.barcode_gap_button.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Error", f"Clustering failed: {e}")

    def visualize_clusters(self, X, labels, num_clusters, dna_texts):
        try:
            # Dimensionality reduction for visualization using PCA
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(X)

            colors = plt.cm.get_cmap("viridis", num_clusters)
            plt.figure(figsize=(10, 6))
            for i in range(num_clusters):
                points = reduced_data[np.array(labels) == i]
                plt.scatter(points[:, 0], points[:, 1], color=colors(i), label=f"Cluster {i+1}")

            # Adding annotations for data points with specific sequence names
            for idx, (point, record) in enumerate(zip(reduced_data, self.sequences)):
                plt.annotate(f"{record.id}", (point[0], point[1]), fontsize=8, alpha=0.6)

            # Adding legend and axis labels
            plt.title("DNA Sequence Clustering Visualization")
            plt.xlabel("PCA Component 1 (Reduced Dimension 1)")
            plt.ylabel("PCA Component 2 (Reduced Dimension 2)")
            plt.legend()
            plt.tight_layout()

            # Save the plot for later export
            self.current_plot = plt.gcf()  # Store the current figure for saving
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Visualization failed: {e}")

    def display_cluster_summary(self, labels, num_clusters):
        try:
            # Clear the text box
            self.cluster_summary_text.delete(1.0, tk.END)

            # Group sequences by cluster
            cluster_groups = {i: [] for i in range(num_clusters)}
            for idx, label in enumerate(labels):
                cluster_groups[label].append(self.sequences[idx].id)

            # Display cluster summary
            self.cluster_summary = ""
            for cluster, sequences in cluster_groups.items():
                self.cluster_summary += f"Cluster {cluster + 1} contains the following sequences:\n"
                for seq_id in sequences:
                    self.cluster_summary += f"- {seq_id}\n"
                self.cluster_summary += "\n"

            self.cluster_summary_text.insert(tk.END, self.cluster_summary)

        except Exception as e:
            messagebox.showerror("Error", f"Displaying cluster summary failed: {e}")

    def barcode_gap_analysis(self):
        try:
            if not hasattr(self, 'distance_matrix') or not hasattr(self, 'cluster_labels'):
                raise ValueError("Please perform clustering before performing Barcode Gap Analysis.")

            distances = self.distance_matrix
            labels = self.cluster_labels
            num_sequences = len(labels)

            # Calculate intra-cluster distances
            intra_distances = []
            for cluster in set(labels):
                indices = [i for i, x in enumerate(labels) if x == cluster]
                cluster_distances = distances[np.ix_(indices, indices)]
                intra_cluster_distances = cluster_distances[np.triu_indices_from(cluster_distances, k=1)]
                intra_distances.extend(intra_cluster_distances)

            # Calculate inter-cluster distances
            inter_distances = []
            clusters = set(labels)
            for cluster_pair in itertools.combinations(clusters, 2):
                indices1 = [i for i, x in enumerate(labels) if x == cluster_pair[0]]
                indices2 = [i for i, x in enumerate(labels) if x == cluster_pair[1]]
                inter_cluster_distances = distances[np.ix_(indices1, indices2)].flatten()
                inter_distances.extend(inter_cluster_distances)

            # Plot histograms of intra-cluster and inter-cluster distances
            plt.figure(figsize=(10, 6))
            plt.hist(intra_distances, bins=30, alpha=0.5, label='Intra-cluster Distances', color='blue', density=True)
            plt.hist(inter_distances, bins=30, alpha=0.5, label='Inter-cluster Distances', color='red', density=True)
            plt.title('Barcode Gap Analysis')
            plt.xlabel('Genetic Distance')
            plt.ylabel('Density')
            plt.legend()
            plt.tight_layout()
            plt.show()

        except Exception as e:
            messagebox.showerror("Error", f"Barcode Gap Analysis failed: {e}")

    def construct_tree(self):
        try:
            # Ensure sequences are loaded and clustered
            if not hasattr(self, 'sequences') or not self.sequences:
                raise ValueError("No sequences loaded. Please load a valid FASTA file.")

            if not hasattr(self, 'distance_matrix') or not hasattr(self, 'cluster_labels'):
                raise ValueError("Please perform clustering before constructing the tree.")

            names = [record.id for record in self.sequences]
            n = len(names)

            # Convert the full distance matrix to a lower triangular format with diagonal zeros
            lower_triangle_matrix = []
            for i in range(n):
                row = []
                for j in range(i + 1):
                    dist = self.distance_matrix[i][j]
                    row.append(dist)
                lower_triangle_matrix.append(row)

            # Create a DistanceMatrix instance
            distance_matrix = DistanceMatrix(names, lower_triangle_matrix)

            # Construct the phylogenetic tree using Neighbor-Joining
            constructor = DistanceTreeConstructor()
            tree = constructor.nj(distance_matrix)

            # Annotate the tree with cluster labels
            self.annotate_tree_with_clusters(tree)

           
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(1, 1, 1)
            Phylo.draw(tree, label_colors=self.label_colors, do_show=False, axes=ax)
            ax.set_title("Phylogenetic Tree with Cluster Annotations")
            plt.show()
            # ---- Modified Section Ends Here ----

            # Save the tree to a file
            save_path = filedialog.asksaveasfilename(defaultextension=".xml",
                                                    filetypes=[("Newick Format", "*.xml"), ("All Files", "*.*")],
                                                    title="Save Phylogenetic Tree")
            if save_path:
                Phylo.write(tree, save_path, "newick")
                messagebox.showinfo("Success", f"Phylogenetic tree saved as {save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"Tree construction failed: {e}")


    def annotate_tree_with_clusters(self, tree):
        # Generate a color map for clusters
        num_clusters = self.num_clusters
        cmap = plt.cm.get_cmap('tab20', num_clusters)
        cluster_colors = [cmap(i) for i in range(num_clusters)]

        # Map sequence names to cluster labels and colors
        self.label_colors = {}
        for idx, record in enumerate(self.sequences):
            cluster_idx = self.cluster_labels[idx]
            color = cluster_colors[cluster_idx]
            self.label_colors[record.id] = color

    def save_results(self):
        try:
            # Save clustering summary
            summary_file = filedialog.asksaveasfilename(defaultextension=".txt",
                                                        filetypes=[("Text Files", "*.txt")],
                                                        title="Save Clustering Summary")
            if summary_file:
                with open(summary_file, 'w') as f:
                    f.write(self.cluster_summary)
                messagebox.showinfo("Success", f"Clustering summary saved as {summary_file}")

            # Save clustering visualization
            plot_file = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG Files", "*.png")],
                                                     title="Save Visualization")
            if plot_file:
                if hasattr(self, 'current_plot'):
                    self.current_plot.savefig(plot_file)
                    messagebox.showinfo("Success", f"Visualization saved as {plot_file}")
                else:
                    messagebox.showwarning("Warning", "No visualization to save.")
        except Exception as e:
            messagebox.showerror("Error", f"Saving results failed: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = DNA_Barcoding_App(root)
    root.mainloop()
