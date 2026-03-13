from util import load_data
sections = ['early', 'middle', 'late']
#iterate through values of k
for i in range (4,11):
    layer_df, cluster_df = load_data(i)
    pitched_cluster = cluster_df[cluster_df['dataset'] != 'drum_loops']
    pitched_layer = layer_df[layer_df['dataset'] != 'drum_loops']
    print("\n\n")
    print(i)
    print(f"\nPITCH Hz responsive neurons:")
    print(f"  Mean per cluster: {pitched_cluster['pitch_responsive'].mean():.1f}")
    print(f"  Max in a cluster: {pitched_cluster['pitch_responsive'].max():.0f}")
    print(f"  Total unique responsive: {pitched_cluster['pitch_responsive'].sum():.0f}")

    print(f"\nBPM responsive neurons:")
    print(f"  Mean per cluster: {cluster_df['bpm_responsive'].mean():.1f}")
    print(f"  Max in a cluster: {cluster_df['bpm_responsive'].max():.0f}")
    print(f"  Total unique responsive: {cluster_df['bpm_responsive'].sum():.0f}")