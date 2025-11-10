import plotly.graph_objects as go


def plot_lmks3d(canonical_lmks, name):
    # Create animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter3d(
                    x=canonical_lmks[frame_idx, :, 0],
                    y=canonical_lmks[frame_idx, :, 1],
                    z=canonical_lmks[frame_idx, :, 2],
                    mode='markers',
                    marker=dict(size=2, color='blue')
                )
            ],
            name=str(frame_idx)
        )
        for frame_idx in range(canonical_lmks.shape[0])
    ]

    # Base figure with first frame
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=canonical_lmks[0, :, 0],
                y=canonical_lmks[0, :, 1],
                z=canonical_lmks[0, :, 2],
                mode='markers',
                marker=dict(size=4, color='blue')
            )
        ],
        layout=go.Layout(
            title=dict(
                text=f'Canonical Landmarks of {name} (3D)',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(title='X'),
                yaxis=dict(title='Y'),
                zaxis=dict(title='Z'),
                aspectmode='data'
            ),
            sliders=[{
                'steps': [
                    {
                        'method': 'animate',
                        'args': [[str(i)], {'mode': 'immediate', 'frame': {'duration': 0}, 'transition': {'duration': 0}}],
                        'label': str(i)
                    } for i in range(canonical_lmks.shape[0])
                ],
                'transition': {'duration': 0},
                'x': 0.1, 'y': -0.1,
                'currentvalue': {'prefix': 'Frame: '}
            }],
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 1.1,
                'x': 1.05,
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }]
        ),
        frames=frames
    )

    fig.show()


def plot_lmks(canonical_lmks, name):
    # Create animation frames
    frames = [
        go.Frame(
            data=[
                go.Scatter(
                    x=canonical_lmks[frame_idx, :, 0],
                    y=canonical_lmks[frame_idx, :, 1],
                    mode='markers',
                    marker=dict(size=4, color='blue')
                )
            ],
            name=str(frame_idx)
        )
        for frame_idx in range(canonical_lmks.shape[0])
    ]

    # Base figure with first frame
    fig = go.Figure(
        data=[
            go.Scatter(
                x=canonical_lmks[0, :, 0],
                y=canonical_lmks[0, :, 1],
                mode='markers',
                marker=dict(size=4, color='blue')
            )
        ],
        layout=go.Layout(
            title=f'Canonical Landmarks of {name}',
            xaxis=dict(title='X'),
            yaxis=dict(title='Y', scaleanchor='x', scaleratio=1),
            sliders=[{
                'steps': [
                    {
                        'method': 'animate',
                        'args': [[str(i)], {'mode': 'immediate', 'frame': {'duration': 0}, 'transition': {'duration': 0}}],
                        'label': str(i)
                    } for i in range(canonical_lmks.shape[0])
                ],
                'transition': {'duration': 0},
                'x': 0.1, 'y': -0.1,
                'currentvalue': {'prefix': 'Frame: '}
            }],
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 1.1,
                'x': 1.05,
                'xanchor': 'right',
                'yanchor': 'top',
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 50, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]
                    }
                ]
            }]
        ),
        frames=frames
    )

    fig.show()