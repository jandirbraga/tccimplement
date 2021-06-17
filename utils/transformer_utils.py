import pandas as pd
import numpy as np

def one_hot_primary_attr(df):
    aux_primary_df = pd.get_dummies(df.primary_attr, prefix='primary_attr')
    df = pd.concat([df, aux_primary_df], axis=1)
    return df

def rename_attack_type(df):
    df.rename(columns={'attack_type':'is_Melle'}, inplace=True)
    df.is_Melle.replace({'Melee': 1, 'Ranged': 0}, inplace=True)
    return df

def one_hot_roles(df):
    distinct_roles = ['Nuker', 'Disabler', 'Initiator', 'Durable', 'Support', 'Jungler', 'Carry', 'Pusher', 'Escape']
    df_aux = pd.DataFrame(df.roles.tolist(), index=df.index).copy()
    df_one_hot_roles = pd.DataFrame(np.zeros([df_aux.shape[0], len(distinct_roles)]), columns=distinct_roles)

    for i in np.arange(df_aux.shape[1]):
        aux_pd = pd.DataFrame()
        for j, role_name in enumerate(distinct_roles):
            df_one_hot_roles[role_name] += (df_aux.loc[:,i] == role_name).astype(int)

    df = pd.concat([df, df_one_hot_roles], axis=1)
    return df


def hero_stats_tranformer(raw_df):
    hero_stats_df = raw_df.copy()
    
    hero_stats_drop_columns = ['roles', 'primary_attr', 'base_health_regen', 'turn_rate', 'name', 'id', 'img',
                           'icon', 'null_win', 'base_health', 'base_mana', 'base_mr', 'cm_enabled']

    hero_stats_df = one_hot_primary_attr(hero_stats_df)
    hero_stats_df = rename_attack_type(hero_stats_df)
    hero_stats_df = one_hot_roles(hero_stats_df)

    hero_stats_df.drop(columns=hero_stats_drop_columns, inplace=True)
    return hero_stats_df