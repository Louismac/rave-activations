from util import load_data
bpm_best = []
pitch_best = []
sections = ['early', 'middle', 'late']
for i in range (4,11):
    layer_df, cluster_df = load_data(i)
    # For pitched datasets only
    pitched_cluster = cluster_df[cluster_df['dataset'] != 'drum_loops']
    pitched_layer = layer_df[layer_df['dataset'] != 'drum_loops']

    for section in sections:
        section_data = cluster_df[cluster_df['section'] == section]
        pitched_section = section_data[section_data['dataset'] != 'drum_loops']
        
        print(f"\n{section.upper()} SECTION:")
        print(f"  Pitch Hz: mean={pitched_section['pitch_mean_corr'].mean():.3f}, max={pitched_section['pitch_mean_corr'].max():.3f}")
        print(f"  BPM:      mean={section_data['bpm_mean_corr'].mean():.3f}, max={section_data['bpm_mean_corr'].max():.3f}")

    # Best section for each feature
    best_pitch_section = pitched_cluster.groupby('section')['pitch_mean_corr'].mean().idxmax()
    pitch_best.append(best_pitch_section)
    best_bpm_section = cluster_df.groupby('section')['bpm_mean_corr'].mean().idxmax()
    bpm_best.append(best_bpm_section)
    print(f"\n🏆 BEST SECTIONS:")
    print(f"  Pitch Hz: {best_pitch_section}")
    print(f"  BPM:      {best_bpm_section}")

print(pitch_best)
print(bpm_best)