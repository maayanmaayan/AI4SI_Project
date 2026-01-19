export type ServiceCategoryId =
  | 'education'
  | 'entertainment'
  | 'grocery'
  | 'health'
  | 'posts_banks'
  | 'parks'
  | 'sustenance'
  | 'shops'

export interface ServiceCategory {
  id: ServiceCategoryId
  label: string
  shortLabel: string
  color: string
  iconEmoji: string
}

export const SERVICE_CATEGORIES: ServiceCategory[] = [
  {
    id: 'education',
    label: 'Education',
    shortLabel: 'Education',
    iconEmoji: '🎓',
    color: '#7BA6FF',
  },
  {
    id: 'entertainment',
    label: 'Entertainment & Culture',
    shortLabel: 'Entertainment',
    iconEmoji: '🎭',
    color: '#FFB3C6',
  },
  {
    id: 'grocery',
    label: 'Everyday Groceries',
    shortLabel: 'Grocery',
    iconEmoji: '🛒',
    color: '#FFD27F',
  },
  {
    id: 'health',
    label: 'Health & Care',
    shortLabel: 'Health',
    iconEmoji: '⚕️',
    color: '#9DE0AD',
  },
  {
    id: 'posts_banks',
    label: 'Posts & Banks',
    shortLabel: 'Posts & Banks',
    iconEmoji: '🏦',
    color: '#C9A6FF',
  },
  {
    id: 'parks',
    label: 'Parks & Open Space',
    shortLabel: 'Parks',
    iconEmoji: '🌳',
    color: '#A3D977',
  },
  {
    id: 'sustenance',
    label: 'Food & Social Life',
    shortLabel: 'Sustenance',
    iconEmoji: '☕️',
    color: '#FFCFBF',
  },
  {
    id: 'shops',
    label: 'Shops & Everyday Needs',
    shortLabel: 'Shops',
    iconEmoji: '🛍️',
    color: '#BFD6FF',
  },
]

