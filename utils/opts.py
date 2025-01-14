def get_args(parser):
    # initial setting
    parser.add_argument('-dataset', type=str, choices=['Extrasensory_dataset', 'ETRI_dataset'], default='Extrasensory_dataset')
    parser.add_argument('-fold', type=int, default=0)
    # training parameter
    parser.add_argument('-epochs', type=int, default=30)
    # model parameter
    # 1. dataloader
    parser.add_argument('-batch_size', type=int, default=512)
    # 2. device setup
    parser.add_argument('--gpu_number', type=int, default=3)
    # 3. optimizer
    parser.add_argument('--optimizer_name', type=str, choices=['AdamW', 'Adam'], default='AdamW')
    parser.add_argument('--lr', type=float, default=0.0001)   # learning weight
    # 4. graph set_up
    parser.add_argument('--self_loop', default=False, action='store_true')
    parser.add_argument('--adjacency_matrix', default=None)
    # 5. model set_up
    parser.add_argument('--num_classes', type=int)
    parser.add_argument('--d_model', type=int, default=128)
    # 5.1 feature_extraction
    parser.add_argument('-motion_ks_1', type=int, default=64)
    parser.add_argument('-motion_ks_2', type=int, default=32)
    parser.add_argument('-motion_stride', type=int, default=2)
    parser.add_argument('-audio_ks_1', type=int, default=8)
    parser.add_argument('-audio_ks_2', type=int, default=6)
    parser.add_argument('-audio_stride', type=int, default=2)
    parser.add_argument('--feature_extractor_activation', type=str, choices=['relu', 'gelu'], default='relu')
    # 5.2 mult model
    parser.add_argument('-mult_heads', type=int, default=4)
    parser.add_argument('-mult_num_layers', type=int, default=2)
    # 5.3 heterogeneous decoder
    parser.add_argument('-dec_num_layers', type=int, default=2)
    parser.add_argument('-dec_heads', type=int, default=2)
    parser.add_argument('-dec_dropout', type=float, default=0.2)
    parser.add_argument('-dec_activation', type=str, choices=['relu', 'gelu'], default='relu')
    parser.add_argument('-dec_layer_norm_eps', type=float, default=1e-5)
    parser.add_argument('-dec_state_kernel', type=int, default=5)
    parser.add_argument('-dec_norm_first', default=True, action='store_true')
    parser.add_argument('-m2l_fusion_type', type=str, choices=['Heter_angleFus', 'angleFus', 'add'], default='Heter_angleFus')
    # 5.4 Heterogeneous GAT
    parser.add_argument('-graph_dropout', type=float, default=0.2)
    parser.add_argument('-leaky_alpha', type=float, default=0.2)
    parser.add_argument('-gat_heads', type=int, default=1)
    parser.add_argument('-graph_regular_lammba', type=float, help='regular index should be [0,1]', default=0.4)
    # 5.5 output
    parser.add_argument('-classifier_bias', default=False, action='store_true')
    # 6. loss
    parser.add_argument('-label_smoothing', type=float, default=0.1)
    parser.add_argument('-loss_regularization', default=True, action='store_true')
    # 7. save_file
    parser.add_argument('-id', type=int, default=0)
    args = parser.parse_args()
    return args