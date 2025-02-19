#!/bin/bash

read -p "Enter expected SHA-1 hashed drive serial number: " EXPECTED_HASH
read -p "Enter ROS bag filename filter (leave empty to skip): " ROSBAG_NAME
read -p "Enter expected SHA-1 hash of ROS bag: " ROSBAG_HASH

echo "Searching for rosbag with the hash $ROSBAG_HASH ($ROSBAG_NAME) on a drive with SN hash $EXPECTED_HASH..."

for drive in /dev/sd?; do
    SERIAL=$(udevadm info --query=property --name="$drive" | grep ID_SERIAL_SHORT= | cut -d= -f2)
    HASHED_SERIAL=$(echo -n "$SERIAL" | sha1sum | awk '{print $1}')

    echo "$drive: SN $SERIAL (hashed $HASHED_SERIAL)"

    if [[ "$HASHED_SERIAL" == "$EXPECTED_HASH" ]]; then
        echo "Match found: $drive"

        partitions=()
        while read -r part fstype; do
            if [[ "$fstype" == "ntfs" || "$fstype" == "ext4" ]]; then
                partitions+=("$part")
            fi
        done < <(lsblk -nr -o PATH,FSTYPE "$drive")

        if [[ ${#partitions[@]} -eq 0 ]]; then
            echo "No NTFS or Ext4 partition found on $drive."
            exit 1
        elif [[ ${#partitions[@]} -gt 1 ]]; then
            echo "Warning: Multiple NTFS/Ext4 partitions found on $drive:"
            printf "%s\n" "${partitions[@]}"
            echo "Using the first one: ${partitions[0]}"
        fi

        selected_partition="${partitions[0]}"
        mountpoint=$(lsblk -nr -o MOUNTPOINT "$selected_partition")

        if [[ -z "$mountpoint" ]]; then
            echo "Partition $selected_partition is not mounted."
            exit 1;
        fi

        echo "Searching for ROS bag files in $mountpoint..."

        SEARCH_FILTER="-name '*.bag'"
        if [[ -n "$ROSBAG_NAME" ]]; then
            SEARCH_FILTER="-name $ROSBAG_NAME"
        fi

        # Using a for loop instead of piping to while to avoid subshell issue
        for bag in $(find "$mountpoint" -type f $SEARCH_FILTER); do
            BAG_HASH=$(sha1sum "$bag" | awk '{print $1}')
            echo "File: $bag (hashed $BAG_HASH)"

            if [[ "$BAG_HASH" == "$ROSBAG_HASH" ]]; then
                echo "Matching ROS bag found: $bag"
                exit 0  # Now this will exit the entire script
            fi
        done

        echo "No matching ROS bag found."
        exit 1  # Exit with failure if no match is found
    fi
done

echo "No matching drive found."
exit 1;
