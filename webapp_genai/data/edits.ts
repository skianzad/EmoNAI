export type Island = {
  id: string;
  label: string;
  mask: string;
  color: string;
  area: number;
};

export type EditLayer = {
  id: string;
  parentId: string | null;
  prompt: string;
  islands: Island[];
};

export const handbag = {
  name: "Handbag",
  original: "/scenes/cafe/original.jpg",
  edited: "/scenes/cafe/edited.jpg",
  layers: [
    {
      id: "cafe",
      parentId: null,
      prompt: "Move the handbag from the chair to the table.",
      islands: [
        {
          id: "bag",
          label: "Handbag moved",
          mask: "/scenes/cafe/bag-move.png",
          color: "#e15b4c",
          area: 36057,
        },
      ],
    },
    {
      id: "cafe-people",
      parentId: null,
      prompt: "Remove the people in the back.",
      islands: [
        {
          id: "people",
          label: "People removed",
          mask: "/scenes/cafe/people.png",
          color: "#2f9d62",
          area: 18080,
        },
      ],
    },
  ] satisfies EditLayer[],
};
